# Trace — prototype walkthrough

Open http://127.0.0.1:5173 in your browser. Use full screen for the presentation.

## A three-minute demonstration

1. Click **Run sample analysis**. The local backend validates and scores 100 synthetic records. The seeded sample has 14 transactions above its illustrative review threshold.
2. Explain the summary counts. Click **All transactions** to filter to flagged records.
3. Click **Inspect** on a transaction. The graph shows its input and output participation; the profile shows its anomaly percentile and observed feature values compared with reference medians.
4. Click **Save to evidence**, then **Evidence** in the sidebar. Enter an observation and click **Save finding & note**.
5. Click **Export report**. The JSON includes source records, hashes, model details, scores, and your saved findings.
6. If time allows, demonstrate **Import data** using `/home/doom/Documents/data/samples/demo.json`. Invalid files show validation errors without replacing the current dataset.

## Explain the scope accurately

The API, validation, database persistence, Isolation Forest scoring, graph responses, filtering, and report generation work locally. The sample and model reference are synthetic. Percentiles are anomaly rankings, not probabilities of crime. Graphs show address participation, not multi-hop UTXO tracing or ownership. Feature comparisons are descriptive deviations, not SHAP explanations. Browser-local analyst notes are included in the exported report.

## Start servers if needed

Backend terminal:

```bash
cd /home/doom/Documents
bash scripts/run-backend.sh
```

Frontend terminal:

```bash
cd /home/doom/bitcoin-intelligence/frontend
export PATH="/home/doom/Documents/.tools/node/bin:$PATH"
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Backend documentation: http://127.0.0.1:8000/docs

Current source files: `src/main.tsx` and `src/styles.css`. The older `BackendAnalysis.tsx` is retained but no longer mounted; the new main workspace integrates its endpoints directly.
