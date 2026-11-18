# Chain Forensics — Elliptic demonstration

## What is implemented

A real supervised classifier trained on the original Elliptic dataset in
`/home/doom/Downloads/archive/elliptic_bitcoin_dataset`.

- 203,769 transaction nodes and 234,355 directed payment edges.
- Input: txId, time_step, 165 anonymized numeric features. ID and time step are excluded from model inputs.
- Labels: original class 1 = illicit (positive), class 2 = licit (negative); unknown is excluded from fitting and metrics.
- Training: known labels in steps 1–29 (26,381 records).
- Validation: steps 30–34 (3,513 known labels); used for candidate selection by Average Precision and threshold selection by validation F1.
- Test: steps 35–49 (16,670 known labels, 50,834 unknown).
- Selected model: histogram gradient boosting. Baseline: logistic regression with training-only scaling.
- Frozen threshold: 0.8176644629613039.

## Measured held-out test performance

| Metric | Value |
|---|---:|
| F1 (illicit class) | 0.8031088083 |
| Precision | 0.9149940968 |
| Recall | 0.7156048015 |
| Average Precision | 0.7962912226 |
| PR-AUC, trapezoidal integration | 0.7962629209 |
| True positives / false positives | 775 / 72 |
| False negatives / true negatives | 308 / 15,515 |

Do not present validation results as test results. Test performance is lower than validation performance, which illustrates why temporal holdout matters.

## Five-minute presentation

1. Open `http://127.0.0.1:5173/`. The default **Elliptic ML** workspace shows the training/validation/test split and test metrics.
2. Show the PR curve and confusion matrix. Explain that 308 illicit transactions were missed; avoid claiming perfect detection.
3. Open **Model card**. Show the two validation candidates and state that the test set was not used to choose the model or threshold.
4. Open **Live evaluation**. Upload `judge-features.csv` only. It returns scores, but evaluation metrics remain unavailable without labels.
5. Add `judge-labels.csv` and score again. Metrics are computed for this uploaded subset, so they need not match the full held-out test metrics.
6. Export predictions and metrics. The export includes dataset hashes, the frozen threshold, and measured request duration.
7. Open **Transaction graph** and inspect a held-out transaction to show published payment-flow neighbors. Node colors represent dataset labels, not inferred identities.

The two sample judge files are in `/home/doom/bitcoin-intelligence/artifacts/elliptic/`.
They contain 200 held-out examples (22 illicit, 178 licit), not a new external dataset.

## If judges give an input

- Original Elliptic-format feature CSV: upload it in Live evaluation (at most 10,000 rows / 32 MiB).
- Separate labels CSV: `txId,class`, with 1, 2, or unknown; labels are matched by ID, not row position.
- Unlabeled records: produce scores/predictions, not invented F1 or PR-AUC.
- Only one class: PR-AUC/AP are withheld and a warning explains the limit.
- Raw Bitcoin address or hash: explain that Elliptic anonymized features cannot be reconstructed by this prototype. Do not promise arbitrary live blockchain ingestion.
- Uploaded records with training/validation IDs: the UI flags overlap, so results are not passed off as independent test performance.

## Reproduce training

```bash
cd /home/doom/bitcoin-intelligence
source /home/doom/Documents/scripts/activate.sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m backend.elliptic --data-dir /home/doom/Downloads/archive/elliptic_bitcoin_dataset
```

Artifacts: `model.joblib`, `benchmark.json`, `predictions.npz`, `edges.csv`, and the two judge sample files in `artifacts/elliptic/`.
Only locally generated model artifacts are loaded; never load a model pickle supplied by an untrusted party.

## Scope and limitations

This is an offline historical Elliptic benchmark. Published aggregate features can contain neighborhood information unavailable at a live decision time; chronological splitting alone does not establish online availability. Labels are the dataset publisher's labels and are not independently re-verified. Scores have not been probability-calibrated. No wallet ownership, identity, or real-time blockchain attribution is claimed.

The **Investigation / Transactions / Evidence** workspace remains a separate local transaction-import and synthetic-reference Isolation Forest demonstration. Its anomaly percentile is not the Elliptic classifier score. Use **Elliptic ML** for the competition's PR-AUC/F1 demonstration.
