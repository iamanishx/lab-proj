"""Train the v2 scam-victim prediction models locally (no notebook needed).

Same pipeline as fraud_detection_v2.ipynb: trains a Logistic Regression
baseline and an MLP (21 -> 32 -> 16 -> 1) on the collected survey/scrape
data, evaluates both on the same held-out split, and saves the models,
encoder and metrics.

Run (from the repo folder, inside your venv):

    python train.py

Optional:

    python train.py --csv data/scam_encounters_india.csv --plots
"""

import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

FEATURES = [
    "channel",
    "claimed_sender",
    "asked_for",
    "urgency_pressure",
    "person_daily_upi_use",
]
SUCCESS = ["lost_money", "shared_otp", "clicked"]


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    for col in FEATURES + ["engaged"]:
        df[col] = df[col].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["engaged"])
    df["scam_worked"] = df["engaged"].isin(SUCCESS).astype(int)
    print("rows:", df.shape)
    print("scam_worked counts:", df["scam_worked"].value_counts().to_dict())
    print("engaged counts:", df["engaged"].value_counts().to_dict())
    return df


def evaluate(name, model, Xte, y_test):
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    m = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    print(name, m)
    return m, proba


def main():
    parser = argparse.ArgumentParser(description="Train the v2 fraud models.")
    parser.add_argument("--csv", default="data/scam_encounters_india.csv",
                        help="input CSV with the same columns as the collected dataset")
    parser.add_argument("--plots", action="store_true",
                        help="also save ROC / EDA plots as PNGs under results/")
    args = parser.parse_args()

    df = load_data(args.csv)

    X = df[FEATURES]
    y = df["scam_worked"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    enc = OneHotEncoder(handle_unknown="ignore")
    Xtr = enc.fit_transform(X_train)
    Xte = enc.transform(X_test)
    print("train/test/positives/features:", len(y_train), len(y_test),
          int(y_test.sum()), Xtr.shape[1])

    lr = LogisticRegression(class_weight="balanced", max_iter=1000,
                            random_state=42).fit(Xtr, y_train)
    m_lr, p_lr = evaluate("LR", lr, Xte, y_test)

    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                        max_iter=300, early_stopping=True,
                        validation_fraction=0.1, random_state=42).fit(Xtr, y_train)
    m_mlp, p_mlp = evaluate("MLP", mlp, Xte, y_test)
    print("MLP epochs:", mlp.n_iter_)

    res = pd.DataFrame({"LR": m_lr, "MLP": m_mlp}).drop(index="confusion")
    print(res)

    names = enc.get_feature_names_out()
    coef = pd.Series(lr.coef_[0], index=names).sort_values()
    print("LR top negative coefficients:\n", coef.head(8))
    print("LR top positive coefficients:\n", coef.tail(8))

    os.makedirs("models_v2", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    joblib.dump(lr, "models_v2/lr_model.joblib")
    joblib.dump(mlp, "models_v2/mlp_model.joblib")
    joblib.dump(enc, "models_v2/encoder.joblib")
    joblib.dump(FEATURES, "models_v2/features.joblib")
    out = {
        "csv": args.csv,
        "rows": int(len(df)),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "test_positive": int(y_test.sum()),
        "lr": m_lr,
        "mlp": m_mlp,
    }
    with open("results/metrics_v2.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved:", os.listdir("models_v2"), "and results/metrics_v2.json")

    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import RocCurveDisplay

        df.groupby("claimed_sender")["scam_worked"].mean().sort_values().plot.barh()
        plt.xlabel("share of attempts that worked")
        plt.tight_layout()
        plt.savefig("results/eda_scam_rate_by_sender.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_test, p_lr, name="LR", ax=ax)
        RocCurveDisplay.from_predictions(y_test, p_mlp, name="MLP", ax=ax)
        fig.savefig("results/roc_curves.png")
        plt.close(fig)

        coef.tail(10).plot.barh()
        plt.tight_layout()
        plt.savefig("results/lr_top_coefficients.png")
        plt.close()
        print("saved plots under results/")

    # demo predictions, same as the notebook's check()
    def check(channel, sender, asked, urgent, upi="unknown"):
        row = pd.DataFrame([[channel, sender, asked, urgent, upi]], columns=FEATURES)
        x = enc.transform(row)
        return (round(float(lr.predict_proba(x)[0, 1]), 3),
                round(float(mlp.predict_proba(x)[0, 1]), 3))

    print("demo (LR, MLP):")
    print(" call/bank_kyc/otp/yes      ->", check("call", "bank_kyc", "otp", "yes"))
    print(" whatsapp/job/money/no      ->", check("whatsapp", "job", "money", "no"))
    print(" sms/prize/link/yes         ->", check("sms", "prize", "link", "yes"))
    print(" email/refund/nothing/no    ->", check("email", "refund", "nothing", "no"))


if __name__ == "__main__":
    main()
