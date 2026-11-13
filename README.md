# Scam outcome prediction: ML mini-project

## Current project: collected data (v2)

Predict a risky-engagement proxy from five categorical fields in 3,005 collected/coded scam-report rows (3,000 public-report rows plus 5 survey responses). `engaged` creates the target; it is not a predictor. A positive label combines clicking, OTP sharing and money loss, so it is not proof every scam succeeded.

- **`explanation.md`**: start here for the dataset, architecture, results, laptop setup and limitations.
- **`fraud_detection_v2.ipynb`**: executed training/evaluation notebook; reads the included `data/scam_encounters_india.csv`.
- **`data/DATASET_NOTES.md`**: provenance and coding conventions. Any approximate noise percentages in these notes are not measured audit results.
- **`results/metrics_v2.json`**: LR F1 **0.7632**, ROC-AUC **0.8372**; MLP F1 **0.7782**, ROC-AUC **0.8482**, on 601 held-out rows.
- **v2.0 model files**: <https://github.com/iamanishx/lab-proj/releases/tag/v2.0> (`lr_model.joblib`, `mlp_model.joblib`, `encoder.joblib`, `features.joblib`).

The notebook uses an 80/20 stratified row split, not a source-grouped split. Related reports may cross train/test. These scores are not a validated estimate of real-world fraud-filter performance.

## Earlier reference: PaySim (v1)

`fraud_detection.ipynb`, `results/metrics.json` and `explanation_v1_paysim.md` preserve the earlier synthetic-transaction experiment. It predicts a different target from different features and must not be treated as a controlled comparison with v2. Its first-run dataset download is separate from the offline-capable v2 workflow.
