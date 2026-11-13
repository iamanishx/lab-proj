# Explanation: the collected-data scam-outcome project (v2)

This is the guide to `fraud_detection_v2.ipynb` and its standalone equivalent `train.py`, updated after the 13 November 2026 run. The old `fraud_detection.ipynb` uses PaySim and remains a separate synthetic-transaction experiment. Its earlier walkthrough is preserved in `explanation_v1_paysim.md`.

## 1. What are we trying to predict?

We study reported scam encounters, not bank transaction logs. Given how an approach arrived, what it claimed and what it asked for, the models predict a binary label called `scam_worked`.

The name is shorthand, not proof that a scam succeeded:

- `1`: the recorded `engaged` outcome is `clicked`, `shared_otp` or `lost_money`.
- `0`: the outcome is `ignored` or `replied`.

This is our chosen **risky-engagement proxy**. Clicking does not necessarily mean losing money, and replying does not necessarily mean being safe. A stronger future study would separate actual financial loss from other actions.

## 2. What is in the dataset?

Classroom rules disallowed Kaggle datasets, so v2 uses a collected and coded CSV: `data/scam_encounters_india.csv`.

| Item | Count |
|---|---:|
| Total rows | 3,005 |
| Coded public-report rows | 3,000 |
| Survey responses | 5 |
| Public rows from ConsumerComplaints.in | 2,229 |
| Public rows from Reddit | 759 |
| Public rows from news sites | 12 |
| Distinct public source URLs | 1,638 |
| Positive label | 1,321 |
| Negative label | 1,684 |

These counts come from the current CSV. They are counts of coded rows, not 3,000 independently verified people or 3,000 unique webpages. Several rows can come from one complaint thread. Public rows record the source in `source_url`; survey rows use the marker `survey`, not a public response URL. The raw reports were not independently fact-checked.

The dataset notes describe manual coding and rule-based coding. They also record approximations: for example, messaging services absent from the schema may be grouped under `whatsapp`, and some identity-document sharing is coded as `shared_otp`. Those choices lose detail and can introduce errors. No measured coding-error percentage is available.

### Five predictors, one outcome, one provenance field

| Column | Role | Values in the current CSV |
|---|---|---|
| `channel` | Predictor | `call`, `sms`, `whatsapp`, `email` |
| `claimed_sender` | Predictor | `bank_kyc`, `refund`, `prize`, `job`, `family_emergency`, `other` |
| `asked_for` | Predictor | `otp`, `pin`, `link`, `qr_scan`, `money`, `nothing` |
| `urgency_pressure` | Predictor | `yes`, `no` |
| `person_daily_upi_use` | Predictor | `unknown`, `1-3 times`, `4+ times` |
| `engaged` | Outcome used to derive the label | `ignored`, `replied`, `clicked`, `shared_otp`, `lost_money` |
| `source_url` | Provenance only | Public URL or `survey` |

`engaged` is **not** a predictor. Feeding it to the model would reveal the answer we are trying to predict. `source_url` is not used as an input either.

Daily UPI use is `unknown` for all 3,000 public-report rows. Only the five survey rows have reported values: three `4+ times` and two `1-3 times`. This is too little evidence for a reliable claim about UPI use. It is not literally guaranteed to have zero effect: the notebook does not include an ablation test that removes this feature and measures the change.

## 3. What does the notebook do?

```text
CSV
  -> trim/lowercase category strings
  -> derive scam_worked from engaged
  -> split rows into train/test, stratified by the label
  -> fit a one-hot encoder on training predictors only
  -> train LR and MLP
  -> evaluate on held-out test rows
  -> save the encoder, models, feature list and metrics
```

### Train/test split

`train_test_split(test_size=0.2, random_state=42, stratify=y)` produces 2,404 training rows and 601 test rows. The test set has 264 positives and 337 negatives. Stratification keeps the label proportions close in both sets; the fixed seed makes this split repeatable.

The split is by **row**, not source URL. Related reports from one thread can cross the split. This limits what the held-out score says about performance on truly new sources.

### One-hot encoding

Models need numbers, not category words. One-hot encoding gives each category its own 0/1 slot: 1 means selected, 0 means not selected. **Five fields do not mean five input numbers.** A field with four possible categories becomes four numbers.

The actual saved v2.0 encoder uses this field and category order:

| Field | Categories, in encoder order | Number of slots |
|---|---|---:|
| `channel` | `call`, `email`, `sms`, `whatsapp` | 4 |
| `claimed_sender` | `bank_kyc`, `family_emergency`, `job`, `other`, `prize`, `refund` | 6 |
| `asked_for` | `link`, `money`, `nothing`, `otp`, `pin`, `qr_scan` | 6 |
| `urgency_pressure` | `no`, `yes` | 2 |
| `person_daily_upi_use` | `1-3 times`, `4+ times`, `unknown` | 3 |
| **Total input numbers per row** | **4 + 6 + 6 + 2 + 3** | **21** |

For example, one scam-attempt row is:

```text
channel = call
claimed_sender = bank_kyc
asked_for = otp
urgency_pressure = yes
person_daily_upi_use = unknown
```

The encoder converts that row into these groups:

```text
call     -> [1, 0, 0, 0]        # call, email, sms, whatsapp
bank_kyc -> [1, 0, 0, 0, 0, 0]  # bank_kyc, family_emergency, job, other, prize, refund
otp      -> [0, 0, 0, 1, 0, 0]  # link, money, nothing, otp, pin, qr_scan
yes      -> [0, 1]              # no, yes
unknown  -> [0, 0, 1]           # 1-3 times, 4+ times, unknown
```

Join the groups, in that order, to get one vector of 21 numbers. There are five 1s in this example, one per field. The 21 is the number of category slots, not the number of original fields, rows or neural layers. For another example, `whatsapp` is `[0, 0, 0, 1]` in this encoder's order.

The encoder learns its categories from **training data only**, then applies the same order to the test data. We did not choose 21 as a neural-network setting. A different training dataset could produce a different category count. `handle_unknown="ignore"` makes an unseen category all-zero within its field; it avoids an error but does not mean the model understands that category.

To inspect it after training in the notebook (or load the saved encoder first):

```python
import joblib

enc = joblib.load("models_v2/encoder.joblib")
features = joblib.load("models_v2/features.joblib")
for field, categories in zip(features, enc.categories_):
    print(field, len(categories), categories.tolist())
print("Total inputs:", sum(len(c) for c in enc.categories_))  # 21
print(enc.get_feature_names_out())  # exact column order
```

### Logistic Regression (LR)

LR learns a weighted combination of the encoded inputs and converts that score to a probability. We use `class_weight="balanced"`, `max_iter=1000` and `random_state=42`. Balanced weighting gives the smaller class more weight in the training objective; it does not guarantee a better test score.

### Multi-layer perceptron (MLP)

The MLP is a small neural network with hidden layers of 32 and 16 units, ReLU activation, `max_iter=300` and `random_state=42`. It can learn nonlinear combinations of the inputs. With `early_stopping=True`, it sets aside 10% of the training data for validation and stops when validation performance no longer improves enough. It stopped after 29 epochs in the saved run. Early stopping helps limit overfitting; it cannot eliminate it.

### What do "Dense 32", "Dense 16" and "ReLU" mean?

**Dense means fully connected.** Every neuron receives all values from the previous layer. Each neuron has its own weights, multiplies the incoming values by those weights, adds the results and adds a bias. One neuron produces one calculated number.

- **Dense 32:** 32 neurons each use all 21 input numbers. They calculate 32 new numbers.
- **ReLU:** replace negative calculated numbers with zero; keep positive numbers unchanged. Zero stays zero. It does not add neurons or change the number of values.
- **Dense 16:** 16 new neurons each use all 32 values from the previous layer. They calculate 16 new numbers, rather than selecting 16 of the old ones.
- **ReLU again:** zero the negatives among those 16 numbers.
- **One output neuron:** use all 16 values to calculate one score. **Sigmoid**, not ReLU, maps that score between 0 and 1.

An illustrative ReLU example (not actual hidden values from the saved model):

```text
Before ReLU: [-2, 3, -0.5, 7]
After ReLU:  [ 0, 3,  0,   7]
```

Four numbers went in and four came out. In our network, the first ReLU keeps 32 values and the second keeps 16.

**32 and 16 are neuron counts we chose, not numbers of layers.** `hidden_layer_sizes=(32, 16)` means two hidden layers: first 32 neurons, then 16. Training learns weights and biases; it does not choose these neuron counts. Together with the final output layer, there are three learned dense layers. The input vector and ReLU operations are not extra groups of neurons.

### Where is ReLU in the notebook code?

There is no separate ReLU import because `MLPClassifier` applies it internally. Our code selects it here:

```python
mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16),  # 32 neurons, then 16 neurons
    activation="relu",           # applied inside both hidden layers
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
).fit(Xtr, y_train)

print(mlp.activation)       # relu: hidden-layer activation
print(mlp.out_activation_)  # logistic: sigmoid for this binary output
```

`activation="relu"` chooses the hidden activation, not the output activation. Scikit-learn uses the logistic (sigmoid) output for this binary classifier. If writing ReLU by hand, `np.maximum(0, z)` does the same zeroing operation; it needs NumPy, not a separate ReLU import.

Read the whole path in plain language:

```text
5 text fields
  -> one-hot encoder: 21 input numbers (4 + 6 + 6 + 2 + 3)
  -> 32 neurons calculate 32 numbers, then ReLU zeros negatives
  -> 16 neurons calculate 16 numbers, then ReLU zeros negatives
  -> 1 neuron calculates 1 score, then sigmoid gives probability p
  -> predict 1 if p >= 0.5, otherwise 0
```

For the exact example row above, the saved v2.0 MLP gives `p = 0.50786631` (about `0.508`), so the 0.5 threshold produces class 1. It is close to the threshold, not a confident warning. The label is risky engagement, and this probability is not a verified chance of losing money.

The saved weight matrices have shapes `(21, 32)`, `(32, 16)` and `(16, 1)`. Including one bias per neuron, their trainable parameter counts are:

```text
First hidden layer:  21 * 32 + 32 = 704
Second hidden layer: 32 * 16 + 16 = 528
Output layer:        16 *  1 +  1 =  17
Total:               704 + 528 + 17 = 1,249
```

### Architecture and why we chose it

The five original input fields become 21 one-hot columns in this run:

```text
LR:  21 encoded inputs -> weighted sum + bias -> positive-label probability
MLP: 21 encoded inputs -> 32 ReLU units -> 16 ReLU units -> positive-label probability
```

LR is an easy-to-inspect baseline. The small MLP is a reasonable starting point for trying interactions among categorical inputs without a large network. These are design reasons, not measured proof that this architecture beats alternatives. We did not compare trees, boosting, larger networks or other hidden-layer sizes, so 32/16 is not shown to be optimal. The MLP win over LR applies only to F1/AUC on this saved split, not every metric or every possible dataset.

## 4. Training results: real held-out test scores

From the executed v2 notebook and `results/metrics_v2.json`, on the same 601 test rows:

| Metric | LR | MLP (32, 16) |
|---|---:|---:|
| Accuracy | 0.7687 | 0.7903 |
| Precision | 0.6935 | 0.7270 |
| Recall | 0.8485 | 0.8371 |
| F1 | 0.7632 | 0.7782 |
| ROC-AUC | 0.8372 | 0.8482 |
| True negatives | 238 | 254 |
| False positives | 99 | 83 |
| False negatives | 40 | 43 |
| True positives | 224 | 221 |

Precision means the share of predicted positives that really have our positive label. Recall means the share of actual positives caught. F1 combines precision and recall. ROC-AUC measures how well the probability scores rank positive rows above negative ones across thresholds; 0.8482 AUC does not mean 84.82% accuracy.

Class predictions use a probability threshold of 0.5. ROC-AUC uses probabilities, not the thresholded predictions. These probabilities have not been checked for calibration, so do not read an output of 0.8 as a proven 80% chance of loss.

The MLP has slightly higher F1, precision, accuracy and ROC-AUC on this split. LR has slightly higher recall. We did not measure uncertainty across repeated splits or establish that the small gap is statistically reliable.

### What do the LR weights mean?

Some coefficients printed by the fitted notebook:

| Encoded input | Coefficient |
|---|---:|
| `asked_for_otp` | +1.4178 |
| `claimed_sender_job` | +1.0391 |
| `channel_whatsapp` | +0.7590 |
| `asked_for_nothing` | -2.0044 |
| `claimed_sender_prize` | -1.4864 |
| `channel_email` | -1.1081 |

Positive coefficients push the fitted LR score toward our positive label, with the other encoded inputs included; negative ones push it the other way. These are fitted associations in this dataset, not causal effects, universal safety rules or percentage-point changes in probability. Full one-hot encoding also means these weights should not be read as simple comparisons against a single omitted reference category.

## 5. How does this compare with PaySim v1?

The earlier notebook predicts a synthetic transaction's `isFraud` label from transaction features. V2 predicts a coded report's engagement proxy from five categorical fields. They are different tasks with different features, class balances and test sets.

| Experiment | LR F1 | LR ROC-AUC | MLP F1 | MLP ROC-AUC |
|---|---:|---:|---:|---:|
| PaySim v1 | 0.5948 | 0.9761 | 0.9585 | 0.9990 |
| Collected-report v2 | 0.7632 | 0.8372 | 0.7782 | 0.8482 |

V2 MLP F1 and both ROC-AUCs are lower, while v2 LR F1 is higher. It would be wrong to say that all results fell, or that this comparison proves real data is harder or proves why scores differ. Synthetic versus collected data is one difference among several, not a controlled explanation. Keep PaySim as a separate reference, not a directly comparable benchmark for this target.

## 6. Run the project on a laptop

The verified run used Python 3.10.12 and scikit-learn 1.7.2. To train without Jupyter, run `python train.py` from the repo root after installing the requirements. Add `--plots` to save graphs. It rewrites local `models_v2/*.joblib` and `results/metrics_v2.json`; see the README for the Windows setup commands.

Internet is needed to clone the repo and install packages initially. After that, v2 training uses the included CSV and does not fetch a dataset. No GPU is needed by this notebook.

### Ubuntu / Linux / macOS terminal

```bash
git clone https://github.com/iamanishx/lab-proj.git
cd lab-proj
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install scikit-learn==1.7.2 pandas numpy matplotlib joblib jupyter
jupyter notebook fraud_detection_v2.ipynb
```

On Ubuntu, if creating the environment reports that `venv` is missing, install the `python3-venv` package using the system package manager, then repeat that step.

### Windows Command Prompt

```bat
git clone https://github.com/iamanishx/lab-proj.git
cd lab-proj
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install scikit-learn==1.7.2 pandas numpy matplotlib joblib jupyter
jupyter notebook fraud_detection_v2.ipynb
```

If already cloned, enter the existing repo and run `git pull` instead of cloning again. Preserve any local edits if Git reports a conflict.

Start Jupyter from the **repo root**, because the first cell has:

```python
CSV = "data/scam_encounters_india.csv"
```

Open the v2 notebook and use **Kernel > Restart & Run All** (wording may vary slightly by Jupyter version). Check that it prints 3,005 rows, 2,404 train and 601 test. The saved run used scikit-learn 1.7.2; other package/platform differences can affect exact results. These commands describe the setup, not a claim that this specific laptop has already passed a fresh run.

Do not run the old PaySim notebook for the collected-data demo: its first-run dataset download is separate and may need internet.

### Using another CSV

Change `CSV` in the first cell. A labelled CSV needs the five predictor columns and `engaged`, with the same category meanings. Validate it before running: `engaged` must be present and one of the five documented outcomes. The current code converts strings before calling `dropna`, and unrecognised outcomes would be treated as negative by `isin`; it is not a complete input-validation pipeline. New categories are tolerated by the encoder, not automatically learned at inference.

## 7. Saved files and release

Running all cells writes:

```text
models_v2/lr_model.joblib
models_v2/mlp_model.joblib
models_v2/encoder.joblib
models_v2/features.joblib
results/metrics_v2.json
```

The notebook saves four model-related files. Use the v2 `encoder.joblib`, not the v1 `scaler.joblib`. The v2.0 release is at:

<https://github.com/iamanishx/lab-proj/releases/tag/v2.0>

The final `check(...)` cell shows how to encode one five-field example and obtain LR/MLP scores. Only load joblib files from trusted sources: they are executable Python serialization, not harmless text.

## 8. Limits and next steps

- Reports are self-selected. Their outcome frequencies do not estimate how often scams work in the general population.
- Coding uses coarse categories and rules. No formal coding-error audit or measured noise rate is available.
- Many rows share source URLs, and identical predictor combinations are common. A random row split is weaker evidence than a source-grouped or independent-source test.
- The label combines actions of different severity. It is not a verified financial-loss label.
- Only five survey responses provide UPI-use information. More survey data is needed before drawing conclusions about that variable.
- Results come from one split and one seed, without confidence intervals, calibration checks or a deployment study.

Next: audit a sample of coded rows, split by source URL, collect broader survey data, separate outcome severities and evaluate on unseen sources. This project demonstrates a re-runnable ML workflow and fitted patterns in collected reports. It is not a bank-ready fraud detector.

## Source files

- `fraud_detection_v2.ipynb`: actual feature list, split, model settings, executed outputs and saved-file paths.
- `results/metrics_v2.json`: held-out v2 metrics.
- `data/scam_encounters_india.csv`: row counts, categories, source-domain counts and label counts.
- `data/DATASET_NOTES.md`: collection/coding conventions and provenance caveats. Its approximate noise percentages are not independently measured and are not treated as results here.
- `fraud_detection.ipynb` and `results/metrics.json`: separate v1 experiment and scores.
