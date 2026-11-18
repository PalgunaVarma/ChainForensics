# Chain Forensics

A local Bitcoin investigation prototype with a trained, chronologically evaluated
Elliptic classifier, precision–recall metrics, feature-file scoring, graph review,
and evidence export.

## Fresh checkout setup

Requires Python 3.12 and a recent Node.js LTS release (tested with Node 24).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm --prefix frontend ci
bash start-chain-forensics.sh
```

The imported-transaction anomaly demonstration works without Elliptic files.
The Elliptic screen reports that training is required until artifacts exist.

## Dataset and model setup

Download the original dataset from [Elliptic’s official Kaggle listing](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
and follow its terms. Dataset CSVs, model binaries, databases, and judge sample
subsets are deliberately excluded from this public repository. The measured
benchmark report is included at `artifacts/elliptic/benchmark.json`.

```bash
source .venv/bin/activate
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m backend.elliptic --data-dir /path/to/elliptic_bitcoin_dataset
```

Training writes the model and two held-out example upload files to
`artifacts/elliptic/`. Restart the backend afterward. The model is never retrained
by judge uploads. Load only model artifacts you generated locally.

## Run on the original demo machine

```bash
cd /home/doom/bitcoin-intelligence
bash start-chain-forensics.sh
```

Open http://127.0.0.1:5173. Keep the terminal open. If the servers are already
running, just open the URL. The script reuses installed runtimes from
`/home/doom/Documents`; fresh checkouts can use the local `.venv` setup above. Set `CHAIN_ACTIVATE_SCRIPT` to use another environment.

## Presentation

See `frontend/ELLIPTIC-DEMO.md`. Use **Elliptic ML** for PR-AUC/F1 demonstrations.
The **Investigation** workspace is a separate synthetic-reference anomaly demo.

Training generates ready-to-upload files in `artifacts/elliptic/` (the original demo machine also has copies in `frontend/demo-input/`):
- `judge-features.csv`: 200 held-out feature rows.
- `judge-labels.csv`: corresponding labels, joined by transaction ID.

## Measured Elliptic test results

Frozen model: histogram gradient boosting. Train steps 1–29, validation 30–34,
test 35–49. Positive class is illicit. Unknown labels are excluded from metrics.
Threshold 0.8176644629613039 was selected using validation F1, never test F1.

- Test known-label count: 16,670 (1,083 illicit; 15,587 licit).
- F1: 0.8031088083.
- Precision: 0.9149940968; recall: 0.7156048015.
- Average Precision: 0.7962912226.
- Trapezoidal PR-AUC: 0.7962629209.
- TP 775, FP 72, FN 308, TN 15,515.

These are offline historical benchmark results, not live-blockchain performance.
The model consumes 165 anonymized features; a raw Bitcoin address alone is not a
valid input. Published aggregate features may contain neighborhood information
unavailable at live decision time. Scores are not calibrated probabilities.

## Files

- `frontend/src/`: React interface.
- `backend/main.py`: transaction import, anomaly scoring, storage, and API.
- `backend/elliptic.py`: chronological training, metrics, and live evaluation.
- `artifacts/elliptic/`: frozen model, dataset hashes, benchmark, predictions, edges.
- `data/samples/`: transaction-import examples for the separate anomaly workspace.

## Retrain

```bash
cd /home/doom/bitcoin-intelligence
source /home/doom/Documents/scripts/activate.sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m backend.elliptic --data-dir /home/doom/Downloads/archive/elliptic_bitcoin_dataset
```

Restart the backend after retraining so it reloads the frozen artifact.

## Verify

```bash
source /home/doom/Documents/scripts/activate.sh
cd /home/doom/bitcoin-intelligence
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest backend -q
npm --prefix frontend run build
# With both servers running:
SATYA_BROWSER_PATH=/home/doom/Documents/.tools/playwright/chromium-1248/chrome-linux64/chrome npm --prefix frontend test
```

## API

- `GET /elliptic/status`: model card and held-out metrics.
- `POST /elliptic/predict`: multipart `features` CSV, optional `labels` CSV.
- `GET /elliptic/transactions`: bounded, paginated test-transaction scores.
- `GET /elliptic/graph/{txid}`: bounded published one-hop neighborhood.
- `GET /openapi.json`: full API schema.

Labels are never used as input features. Live requests never retrain or tune the
model. Missing labels yield no evaluation metrics; single-class inputs withhold
PR-AUC/AP. Training/validation ID overlaps are reported. The JSON export includes
file hashes, predictions, threshold, and measured inference duration.

## References

- [Elliptic dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
- [Weber et al. (2019), Anti-Money Laundering in Bitcoin](https://arxiv.org/abs/1908.02591)
- [scikit-learn gradient boosting](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting)
- [FastAPI](https://fastapi.tiangolo.com/), [React](https://react.dev/), [Vite](https://vite.dev/)

This implementation uses a tabular classifier; it does not reproduce the paper’s GCN.
