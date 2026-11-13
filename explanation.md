# Explanation: how this fraud detector works, from zero

Written for someone who has never trained an ML model. Read top to bottom and
you'll understand every part of the project.

## The problem in one sentence

A mobile-money app moves money between accounts; a tiny fraction of transfers
are criminals emptying someone's account, and we want a program that looks at
each transaction and says **"fraud"** or **"legit"** - fast, while transactions
are still happening.

```
                        +---------------------------+
  new transaction       |       trained model       |      fraud-prob 0.97
  --------------------> |  (logistic reg. or MLP)   | ------------------> ALERT if >= 0.5
  type, amount,         |                           |
  balances before/after +------------+--------------+
                                     |
                          learned from 6.3M past
                          transactions (PaySim)
```

## Step 0: the data, in detail

**What we used:** the **PaySim** dataset - "Synthetic Financial Datasets For
Fraud Detection", the standard public dataset for mobile-money fraud work.
Real transaction logs are private financial data that no company can publish,
so researchers built PaySim: a simulator calibrated on aggregated stats from
real logs of a mobile-money service running in 14+ countries. It behaves like
the real thing, including injected criminals.

- Source: <https://www.kaggle.com/datasets/ealaxi/paysim1> (the notebook
  downloads the identical CSV from a public Hugging Face mirror,
  <https://huggingface.co/datasets/theman10/paysim>, because Kaggle needs a
  login)
- Paper: Lopez-Rojas, Elmir, Axelsson, "PaySim: A financial mobile money
  simulator for fraud detection", EMSS 2016.

**How big it is:**

| fact | value |
|---|---|
| file size | ~470 MB CSV |
| total transactions | **6,362,620** |
| time span | 744 hourly steps = one simulated month |
| fraud transactions | **8,213 (0.13%)** |
| legit transactions | 6,354,407 (99.87%) |

**What's in each row:** `step` (which hour), `type` (one of 5 kinds), `amount`,
`nameOrig`/`nameDest` (sender/receiver IDs), `oldbalanceOrg`/`newbalanceOrig`
(sender balance before/after), `oldbalanceDest`/`newbalanceDest` (receiver
before/after), `isFraud` (the answer), `isFlaggedFraud` (the bank's old rule).

**The 5 transaction types, counted from the actual file:**

| type | rows | fraud rows |
|---|---:|---:|
| CASH_OUT | 2,237,500 | **4,116** |
| PAYMENT | 2,151,495 | 0 |
| CASH_IN | 1,399,284 | 0 |
| TRANSFER | 532,909 | **4,097** |
| DEBIT | 41,432 | 0 |

Fraud happens **only** in TRANSFER and CASH_OUT - the criminal moves money out
of the victim's account, then withdraws it. That's why we train only on those
two types.

That imbalance is the whole game. A "model" that says *legit* every single time
is already 99.87% accurate - and completely useless. So we never trust plain
accuracy; we watch **precision** and **recall** (defined below).

## Step 1: keep the slice where fraud lives

Each transaction has a type: PAYMENT, CASH_IN, CASH_OUT, TRANSFER, DEBIT.
In PaySim, fraud **only** ever appears as TRANSFER or CASH_OUT - the criminal
moves money out of the victim's account (TRANSFER) and then withdraws it
(CASH_OUT). So we train only on those two types.

**The exact training subset:** TRANSFER + CASH_OUT together are 2,770,409
rows. Training on all of them would be slow for no real gain (they're 99.7%
legit), so we keep **100% of the 8,213 fraud rows** and a random **5% of the
legit rows** - **145,861 rows total** (8,213 fraud + 137,648 legit). The 5%
sampling uses a fixed random seed, so rerunning gives the identical subset. The
notebook prints these counts when it runs, so the subset is stated, not hidden.
Why this is fair: every fraud case is in there, and the legit sample is random,
so the models still see the real variety of honest transactions - just without
making you wait all day.

## Step 2: features - what the model is allowed to see

The model doesn't see account names (random IDs teach nothing). It sees 8
numbers per transaction:

- the **type** (TRANSFER or not), the **amount**,
- sender balance **before/after**, receiver balance **before/after**,
- two hand-built **balance-error** features:
  `errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrig` (and the same
  on the destination side). For an honest transaction this is 0 - the books
  balance. Fraudulent transfers often leave a mismatch, and the models pick up
  on it heavily.

Before training, all features are **standardized** (shifted to mean 0, spread
1) so that "amount in the millions" doesn't drown out "error of 0.01".

## Step 3: model #1 - logistic regression (the baseline)

Logistic regression draws **one straight boundary** through the 8-dimensional
feature space: high `amount` + zeroed balance pushes a transaction over the
line. It's the simplest model that can work, so we train it first as the bar
everything else must clear. `class_weight="balanced"` makes it care about the
rare fraud class instead of ignoring it.

## Step 4: model #2 - a small MLP (neural network)

An MLP (multi-layer perceptron) stacks simple units that each re-mix the
features, so it can learn **combinations**: "big amount *and* balance emptied
*and* it's a transfer". Ours has two hidden layers of 32 and 16 neurons - small
on purpose, because the dataset is tabular and small nets train in seconds and
don't overfit. `early_stopping` halts training automatically when it stops
improving.

## Step 5: judging honestly

We hide 20% of the data from training (a **test set**: 29,173 transactions,
1,643 of them fraud) and grade the models only on that.

- **Precision**: of everything we flagged as fraud, how much really was?
  (Low precision = analysts drowning in false alarms.)
- **Recall**: of all the real fraud, how much did we catch?
  (Low recall = criminals walking away with money.)
- **AUPRC**: area under the precision-recall curve - the fairest single score
  when one class is this rare.

Real numbers from the executed notebook (test set):

| metric | logistic regression | MLP (32,16) |
|---|---|---|
| accuracy | 0.9322 | 0.9954 |
| precision | 0.4484 | 0.9689 |
| recall | 0.8831 | 0.9483 |
| F1 | 0.5948 | 0.9585 |
| ROC-AUC | 0.9761 | 0.9990 |
| AUPRC | 0.8287 | 0.9927 |

Reading the story: the linear baseline catches most fraud (recall 0.88) but
**half of its alarms are false** (precision 0.45). The MLP catches even more of it (recall 0.95) while cutting false alarms ~35x (precision 0.97). Its
confusion matrix on 29,173 test transactions: 50 false alarms, 85 frauds
missed, 1,558 frauds caught.

## Step 6: the "real-time" part

A batch report at the end of the day doesn't stop a theft in progress. So the
last part of the notebook replays the raw log **in time order** and scores each
transaction one at a time, as if it were arriving live - the demo prints each
alert with its fraud probability. Measured on a plain CPU it scored **~6,800
transactions per second**, thousands of times faster than any real payment
rail, so the model can sit inside a live pipeline.

## Run it yourself

Local:

```bash
pip install -r requirements.txt
jupyter notebook fraud_detection.ipynb   # Run All - downloads the data itself
```

Colab (free): upload `fraud_detection.ipynb` to <https://colab.research.google.com>,
Runtime > Run all. CPU is fine - the full run is under a minute; a GPU changes
nothing for models this small.

The trained weights (`models/*.joblib`) are on the GitHub release, so you can
also skip training and just load them:

```python
import joblib
mlp = joblib.load("mlp_model.joblib")
scaler = joblib.load("scaler.joblib")
# fraud-prob = mlp.predict_proba(scaler.transform([features]))[0, 1]
```

## Where everything was computed

Every number, plot and the stream demo above came from actually executing
`fraud_detection.ipynb` once on a CPU-only Linux sandbox (Python 3.10,
scikit-learn 1.7.2) on Nov 13, 2026. No GPU was used or needed. The
`results/metrics.json` file in this repo is the exact output of that run.
