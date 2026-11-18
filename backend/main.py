"""Offline demo API. All amounts are integer satoshis; network IPs are relays."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import numpy as np
from defusedxml import ElementTree
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    IPvAnyAddress,
    ValidationError,
    field_validator,
    model_validator,
)
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, confusion_matrix, precision_recall_fscore_support
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 2 * 1024 * 1024
MAX_ROWS = 2000
MODEL_VERSION = "isolation-forest-synthetic-reference-v1"
FEATURES = [
    "log10_total_input_sats",
    "input_count",
    "output_count",
    "fee_fraction",
    "largest_output_share",
]
DISCLAIMER = "Demo anomaly detection against a synthetic reference. Scores are not criminality probabilities. IPs indicate observed relay endpoints, not ownership. Address strings are not Bitcoin-checksum validated."
Money = Annotated[int, Field(strict=True, ge=0, le=2_100_000_000_000_000)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: datetime | None = None
    src_ip: IPvAnyAddress | None = None
    dst_ip: IPvAnyAddress | None = None
    src_port: int | None = Field(default=None, ge=1, le=65535)
    dst_port: int | None = Field(default=None, ge=1, le=65535)
    txid: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    input_addresses: list[str] = Field(min_length=1, max_length=100)
    output_addresses: list[str] = Field(min_length=1, max_length=100)
    input_amounts: list[Money] = Field(min_length=1, max_length=100)
    output_amounts: list[Money] = Field(min_length=1, max_length=100)
    fee: Money | None = None
    script_type: str = Field(default="unknown", max_length=80)
    geo_country: str | None = Field(default=None, max_length=2)
    asn: int | None = Field(default=None, ge=0, le=4294967295)

    label: Annotated[int, Field(strict=True, ge=0, le=1)] | None = None

    @field_validator("timestamp")
    @classmethod
    def utc_timestamp(cls, value):
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        return value.astimezone(UTC)

    @field_validator("txid")
    @classmethod
    def lower_txid(cls, value):
        return value.lower()

    @field_validator("input_addresses", "output_addresses")
    @classmethod
    def valid_addresses(cls, values):
        if any(not value.strip() or len(value) > 128 for value in values):
            raise ValueError("addresses must be nonblank and at most 128 characters")
        return values

    @model_validator(mode="after")
    def balances(self):
        if len(self.input_addresses) != len(self.input_amounts) or len(
            self.output_addresses
        ) != len(self.output_amounts):
            raise ValueError("address and amount arrays must have matching lengths")
        total = sum(self.input_amounts)
        delta = total - sum(self.output_amounts)
        if total <= 0 or total > 2_100_000_000_000_000 or delta < 0:
            raise ValueError("invalid transaction value conservation")
        if self.fee is not None and self.fee != delta:
            raise ValueError("fee does not match inputs minus outputs")
        self.fee = delta
        return self


class DemoRequest(BaseModel):
    count: int = Field(default=100, ge=10, le=1000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


def feature_vector(row):
    total = sum(row["input_amounts"])
    output_total = sum(row["output_amounts"])
    return [
        np.log10(1 + total),
        len(row["input_addresses"]),
        len(row["output_addresses"]),
        row["fee"] / total,
        max(row["output_amounts"]) / max(output_total, 1),
    ]


def generate_records(count, seed, unusual=False):
    """Independent balanced mock transactions, NOT a complete UTXO simulator."""
    rng = np.random.default_rng(seed)
    result = []
    for index in range(count):
        outlier = unusual and index % 10 == 0
        inputs = int(rng.integers(1, 4))
        outputs = int(rng.integers(15, 35) if outlier else rng.integers(1, 5))
        total = int(
            rng.integers(10_000_000_000, 90_000_000_000)
            if outlier
            else rng.integers(100_000, 50_000_000)
        )
        fee = (
            int(total * rng.uniform(0.05, 0.12))
            if outlier
            else int(rng.integers(100, 5000))
        )

        def split(amount, size):
            values = [amount // size] * size
            values[-1] += amount % size
            return values

        record = Record(
            timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index),
            src_ip=f"192.0.2.{index % 250 + 1}",
            dst_ip="198.51.100.1",
            src_port=40000 + index % 1000,
            dst_port=8333,
            txid=hashlib.sha256(f"{seed}:{index}:{outlier}".encode()).hexdigest(),
            input_addresses=[f"demo_{seed}_{index}_in_{i}" for i in range(inputs)],
            output_addresses=[f"demo_{seed}_{index}_out_{i}" for i in range(outputs)],
            input_amounts=split(total, inputs),
            output_amounts=split(total - fee, outputs),
            fee=fee,
            script_type="synthetic",
            geo_country=None,
            asn=None,
        )
        result.append(record.model_dump(mode="json"))
    return result


@lru_cache(maxsize=1)
def detector():
    reference = np.asarray([feature_vector(row) for row in generate_records(600, 9182)])
    calibration = np.asarray(
        [feature_vector(row) for row in generate_records(300, 3817)]
    )
    model = IsolationForest(n_estimators=100, random_state=42, n_jobs=1).fit(reference)
    scores = np.sort(-model.score_samples(calibration))
    return (
        model,
        scores,
        np.median(reference, axis=0),
        np.maximum(reference.std(axis=0), 1e-6),
    )


def score(row):
    model, reference_scores, median, spread = detector()
    vector = np.asarray(feature_vector(row))
    anomaly = float(-model.score_samples(vector.reshape(1, -1))[0])
    percentile = round(
        float(
            np.searchsorted(reference_scores, anomaly, side="right")
            / len(reference_scores)
            * 100
        ),
        2,
    )
    deviations = np.abs((vector - median) / spread)
    reasons = [
        {
            "feature": FEATURES[int(i)],
            "value": round(float(vector[i]), 6),
            "reference_median": round(float(median[i]), 6),
        }
        for i in np.argsort(deviations)[-3:][::-1]
    ]
    return {
        "txid": row["txid"],
        "anomaly_score": anomaly,
        "anomaly_percentile": percentile,
        "flagged": percentile >= 95,
        "reasons": reasons,
        "explanation_method": "Largest standardized deviations from reference; descriptive context, not feature attribution.",
        "model_version": MODEL_VERSION,
    }


def connect():
    path = Path(os.environ.get("CHAIN_FORENSICS_DB_PATH", str(ROOT / "data" / "demo.sqlite3")))
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.execute(
        "CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
    )
    return db


def load_dataset(dataset_id):
    db = connect()
    try:
        row = db.execute(
            "SELECT payload FROM datasets WHERE id=?", (dataset_id,)
        ).fetchone()
    finally:
        db.close()
    if row is None:
        raise HTTPException(404, "Dataset not found")
    return json.loads(row[0])


def summary(dataset):
    return {
        key: value
        for key, value in dataset.items()
        if key not in ("transactions", "alerts", "observations", "labels")
    }


def ingest(rows, source_name, source_hash, synthetic=False):
    started = perf_counter()
    if not isinstance(rows, list) or not rows:
        raise HTTPException(422, "Input must contain a nonempty list of records")
    if len(rows) > MAX_ROWS:
        raise HTTPException(413, f"Maximum {MAX_ROWS} records per import")
    transactions, observations, errors, seen = {}, [], [], set()
    duplicates = 0
    for index, raw in enumerate(rows, start=1):
        try:
            row = Record.model_validate(raw).model_dump(mode="json")
            body = {
                key: row[key]
                for key in (
                    "txid",
                    "input_addresses",
                    "output_addresses",
                    "input_amounts",
                    "output_amounts",
                    "fee",
                    "script_type",
                )
            }
            if (
                row["txid"] in transactions
                and transactions[row["txid"]]["body"] != body
            ):
                raise ValueError("conflicting blockchain fields for the same txid")
            fingerprint = json.dumps(row, sort_keys=True)
            if fingerprint in seen:
                duplicates += 1
                continue
            seen.add(fingerprint)
            transactions.setdefault(
                row["txid"], {"body": body, "record": row, "source_records": []}
            )["source_records"].append(index)
            observations.append({"source_record": index, **row})
        except (ValidationError, ValueError) as error:
            if isinstance(error, ValidationError):
                message = "; ".join(
                    f"{'.'.join(map(str, item['loc']))}: {item['msg']}"
                    for item in error.errors()
                )
            else:
                message = str(error)
            errors.append({"source_record": index, "error": message})
    if not transactions:
        raise HTTPException(
            422, {"message": "No valid transactions", "errors": errors[:100]}
        )
    # A label never participates in features or model fitting.
    labels = {}
    for observation in observations:
        label = observation.get("label")
        if label is not None:
            key = observation["txid"]
            if key in labels and labels[key] != label:
                raise HTTPException(422, "Conflicting labels for the same transaction; import rejected")
            labels[key] = label
    items = list(transactions.values())
    alerts = score_batch([item["record"] for item in items])
    for alert, item in zip(alerts, items):
        alert["source_records"] = item["source_records"]
    alerts.sort(key=lambda alert: alert["anomaly_score"], reverse=True)
    dataset = {
        "id": str(uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "source_name": source_name,
        "source_sha256": source_hash,
        "synthetic": synthetic,
        "amount_unit": "satoshi",
        "labels": labels,
        "label_source": "synthetic-generator" if synthetic else "user-supplied-unverified",
        "positive_class": "generator-injected anomaly" if synthetic else "user-defined positive class (label 1)",
        "analysis_ms": round((perf_counter() - started) * 1000, 2),
        "decision_threshold_percentile": 95,
        "training_reference": {"records": 600, "seed": 9182, "synthetic": True},
        "calibration_reference": {"records": 300, "seed": 3817, "synthetic": True},
        "model_version": MODEL_VERSION,
        "disclaimer": DISCLAIMER,
        "input_rows": len(rows),
        "accepted_observations": len(observations),
        "unique_transactions": len(transactions),
        "duplicates": duplicates,
        "rejected_count": len(errors),
        "errors": errors[:100],
        "flagged_count": sum(alert["flagged"] for alert in alerts),
        "transactions": transactions,
        "observations": observations,
        "alerts": alerts,
    }
    db = connect()
    try:
        with db:
            db.execute(
                "INSERT INTO datasets VALUES (?, ?)",
                (dataset["id"], json.dumps(dataset)),
            )
    finally:
        db.close()
    return summary(dataset)


app = FastAPI(
    title="Chain Forensics API",
    version="0.1.0",
    description=DISCLAIMER,
    docs_url=None,
    redoc_url=None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health():
    return {"status": "ok", "mode": "offline-demo", "model_version": MODEL_VERSION}


@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def docs():
    # No CDN assets: basic interactive documentation works disconnected.
    return """<!doctype html><html><head><title>Chain Forensics API</title><meta charset="utf-8"></head>
    <body style="font:16px system-ui;max-width:960px;margin:40px auto;padding:20px">
    <h1>Chain Forensics backend</h1><p>Synthetic-reference anomaly detection. Not criminality probabilities.</p>
    <button id="run">Generate and analyze 100 demo transactions</button>
    <p id="links"></p><pre id="result" style="white-space:pre-wrap"></pre>
    <h2>API endpoints</h2><ul><li>GET /health</li><li>POST /demo</li><li>POST /datasets (multipart file)</li>
    <li>GET /datasets</li><li>GET /datasets/{id}</li><li>GET /datasets/{id}/alerts</li>
    <li>GET /datasets/{id}/graph</li><li>GET /datasets/{id}/export</li></ul>
    <a href="/openapi.json">OpenAPI schema</a><script>
    document.getElementById('run').onclick=async()=>{const b=document.getElementById('run');b.disabled=true;
    try{const r=await fetch('/demo',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({count:100,seed:42})});
    const d=await r.json();document.getElementById('result').textContent=JSON.stringify(d,null,2);
    const p=document.getElementById('links');p.replaceChildren();if(r.ok)for(const name of ['alerts','graph','export']){
    const a=document.createElement('a');a.href='/datasets/'+encodeURIComponent(d.id)+'/'+name;a.textContent=name+' ';p.append(a);}}
    catch(e){document.getElementById('result').textContent=String(e)}finally{b.disabled=false}};</script></body></html>"""


@app.post("/demo", status_code=201)
def demo(request: DemoRequest):
    if request.seed in (9182, 3817):
        raise HTTPException(422, "Choose a seed other than the training and calibration seeds")
    rows = generate_records(request.count, request.seed, unusual=True)
    for index, row in enumerate(rows):
        row["label"] = int(index % 10 == 0)
    return ingest(
        rows,
        f"synthetic-seed-{request.seed}.json",
        hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
        True,
    )


@app.post("/datasets", status_code=201)
def upload(file: Annotated[UploadFile, File()]):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File limit is 2 MiB")
    suffix = Path(file.filename or "").suffix.lower()
    try:
        text = data.decode("utf-8-sig")
        if suffix == ".json":
            rows = json.loads(text)
        elif suffix == ".csv":
            rows = list(csv.DictReader(io.StringIO(text)))
            arrays = (
                "input_addresses",
                "output_addresses",
                "input_amounts",
                "output_amounts",
            )
            for row in rows:
                for key in arrays:
                    row[key] = json.loads(row[key])
                for key in ("fee", "src_port", "dst_port", "asn", "label"):
                    if key in row:
                        row[key] = int(row[key]) if row[key] else None
                if row.get("geo_country") == "":
                    row["geo_country"] = None
        elif suffix == ".xml":
            root = ElementTree.fromstring(text)
            if root.tag != "records":
                raise ValueError("XML root must be <records>")
            rows = []
            for element in root:
                if element.tag != "record":
                    raise ValueError("XML children must be <record>")
                row = {child.tag: child.text or "" for child in element}
                for key in (
                    "input_addresses",
                    "output_addresses",
                    "input_amounts",
                    "output_amounts",
                ):
                    row[key] = json.loads(row[key])
                for key in ("fee", "src_port", "dst_port", "asn", "label"):
                    if key in row:
                        row[key] = int(row[key]) if row[key] else None
                if row.get("geo_country") == "":
                    row["geo_country"] = None
                rows.append(row)
        else:
            raise HTTPException(415, "Use a .json, .csv, or .xml file")
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            422, "Malformed input file; check the documented format"
        ) from error
    return ingest(
        rows, Path(file.filename or "upload").name, hashlib.sha256(data).hexdigest()
    )


@app.get("/datasets")
def datasets(limit: int = Query(20, ge=1, le=100)):
    db = connect()
    try:
        rows = db.execute(
            "SELECT payload FROM datasets ORDER BY rowid DESC LIMIT ?", (limit,)
        ).fetchall()
    finally:
        db.close()
    return [summary(json.loads(row[0])) for row in rows]


@app.get("/datasets/{dataset_id}")
def dataset(dataset_id: str):
    return summary(load_dataset(dataset_id))


@app.get("/datasets/{dataset_id}/alerts")
def alerts(
    dataset_id: str, flagged_only: bool = True, limit: int = Query(50, ge=1, le=200)
):
    data = load_dataset(dataset_id)
    return {
        "dataset_id": dataset_id,
        "disclaimer": DISCLAIMER,
        "alerts": [a for a in data["alerts"] if not flagged_only or a["flagged"]][
            :limit
        ],
    }


@app.get("/datasets/{dataset_id}/graph")
def graph(
    dataset_id: str, txid: str | None = None, limit: int = Query(20, ge=1, le=50)
):
    data = load_dataset(dataset_id)
    selected = [txid.lower()] if txid else [a["txid"] for a in data["alerts"][:limit]]
    nodes, edges = {}, []
    for key in selected:
        if key not in data["transactions"]:
            raise HTTPException(404, "Transaction not found")
        row = data["transactions"][key]["record"]
        tid = f"tx:{key}"
        nodes[tid] = {"id": tid, "type": "transaction", "label": key}
        for side in ("input", "output"):
            for index, (address, amount) in enumerate(
                zip(row[f"{side}_addresses"], row[f"{side}_amounts"])
            ):
                aid = f"address:{address}"
                nodes[aid] = {"id": aid, "type": "address", "label": address}
                edges.append(
                    {
                        "id": f"{tid}:{side}:{index}",
                        "source": aid if side == "input" else tid,
                        "target": tid if side == "input" else aid,
                        "type": side,
                        "amount_sats": amount,
                        "txid": key,
                    }
                )
    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "limitations": "Address participation only. No UTXO spend paths or ownership inference.",
        "truncated": txid is None and len(data["transactions"]) > limit,
    }


@app.get("/datasets/{dataset_id}/export")
def export(dataset_id: str):
    data = load_dataset(dataset_id)
    return {**data, "evaluation": evaluate_dataset(data)}


def score_batch(rows):
    """Vectorized inference using the fixed synthetic reference model."""
    model, reference_scores, median, spread = detector()
    matrix = np.asarray([feature_vector(row) for row in rows])
    raw = -model.score_samples(matrix)
    percentiles = np.searchsorted(reference_scores, raw, side="right") / len(reference_scores) * 100
    results = []
    for row, vector, anomaly, percentile in zip(rows, matrix, raw, percentiles):
        deviations = np.abs((vector - median) / spread)
        results.append({
            "txid": row["txid"], "anomaly_score": float(anomaly),
            "anomaly_percentile": round(float(percentile), 2), "flagged": bool(percentile >= 95),
            "reasons": [{"feature": FEATURES[int(i)], "value": round(float(vector[i]), 6),
                         "reference_median": round(float(median[i]), 6)} for i in np.argsort(deviations)[-3:][::-1]],
            "explanation_method": "Largest standardized deviations from reference; descriptive context, not feature attribution.",
            "model_version": MODEL_VERSION,
        })
    return results


def evaluate_dataset(data):
    labels = data.get("labels", {})
    matched = [a for a in data["alerts"] if a["txid"] in labels]
    base = {
        "dataset_id": data["id"], "model_version": data["model_version"],
        "source_sha256": data["source_sha256"], "label_source": data.get("label_source", "not-provided"),
        "positive_class": data.get("positive_class", "user-defined positive class (label 1)"),
        "synthetic": data["synthetic"], "total_transactions": len(data["alerts"]),
        "labeled_transactions": len(matched), "unlabeled_transactions": len(data["alerts"])-len(matched),
        "label_coverage": len(matched)/len(data["alerts"]) if data["alerts"] else 0,
        "threshold": {"field": "anomaly_percentile", "operator": ">=", "value": 95, "policy": "Fixed before evaluation; not tuned on this dataset"},
        "score_used_for_pr_curve": "raw anomaly_score (higher means more anomalous)",
        "precision": None, "recall": None, "f1": None, "average_precision": None,
        "pr_auc_trapezoidal": None, "confusion_matrix": None, "curve": [], "warnings": [],
    }
    if not matched:
        return {**base, "status": "labels_required", "warnings": ["Ground-truth labels are required. Predictions cannot be used as ground truth."]}
    y = np.array([labels[a["txid"]] for a in matched])
    predictions = np.array([int(a["flagged"]) for a in matched])
    scores = np.array([a["anomaly_score"] for a in matched])
    precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0,1]).ravel()
    base.update(status="evaluated", positives=int(y.sum()), negatives=int(len(y)-y.sum()),
                precision=float(precision), recall=float(recall), f1=float(f1),
                confusion_matrix={"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)},
                prevalence=float(y.mean()))
    if len(np.unique(y)) == 2:
        p, r, _ = precision_recall_curve(y, scores)
        base.update(average_precision=float(average_precision_score(y,scores)),
                    pr_auc_trapezoidal=float(auc(r,p)),
                    curve=[{"recall":float(x),"precision":float(v)} for x,v in zip(r,p)])
    else:
        base["warnings"].append("Only one class is labeled. PR-AUC and AP are withheld; F1 is not a representative two-class benchmark.")
    if not predictions.any():
        base["warnings"].append("No predicted positives: precision is undefined and displayed as 0 by the stated zero-division convention.")
    if not y.any():
        base["warnings"].append("No positive ground-truth examples: recall is undefined and displayed as 0.")
    if len(matched) < len(data["alerts"]):
        base["warnings"].append("Metrics use labeled transactions only. Unlabeled records are excluded, not treated as negatives.")
    if data["synthetic"]:
        base["warnings"].append("Synthetic generator benchmark only; this does not establish real-world illicit-activity detection performance.")
    else:
        base["warnings"].append("Labels are supplied by the uploader and not independently verified. Model training remains synthetic.")
    return base


@app.get("/datasets/{dataset_id}/evaluation")
def evaluation(dataset_id: str):
    return evaluate_dataset(load_dataset(dataset_id))


@app.post("/datasets/{dataset_id}/labels")
def attach_labels(dataset_id: str, file: Annotated[UploadFile, File()]):
    data = load_dataset(dataset_id)
    raw = file.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Labels file limit is 2 MiB")
    try:
        text = raw.decode("utf-8-sig")
        suffix = Path(file.filename or "").suffix.lower()
        if suffix == ".json":
            rows = json.loads(text)
        elif suffix == ".csv":
            rows = list(csv.DictReader(io.StringIO(text)))
            for row in rows:
                if row.get("label") not in ("0", "1"):
                    raise ValueError("CSV labels must be exactly 0 or 1")
                row["label"] = int(row["label"])
        else:
            raise ValueError("Use JSON or CSV labels")
        if not isinstance(rows,list) or not rows or len(rows)>MAX_ROWS:
            raise ValueError("Provide 1 to 2,000 label rows")
        labels = {}
        for row in rows:
            if not isinstance(row,dict) or set(row) != {"txid", "label"}:
                raise ValueError("Each row must contain exactly txid and label")
            if not isinstance(row["txid"], str):
                raise ValueError("txid must be a string")
            key = row["txid"].lower()
            if key not in data["transactions"]:
                raise ValueError(f"Unknown transaction ID: {key}")
            if type(row["label"]) is not int or row["label"] not in (0,1):
                raise ValueError("Labels must be integer 0 or 1; unknown labels must be omitted")
            if key in labels:
                raise ValueError("Duplicate transaction ID in label file")
            labels[key] = row["label"]
    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(422, str(error)) from error
    data["labels"] = labels
    data["label_source"] = "user-supplied-unverified"
    data["positive_class"] = "user-defined positive class (label 1)"
    data["labels_sha256"] = hashlib.sha256(raw).hexdigest()
    data["labels_updated_at"] = datetime.now(UTC).isoformat()
    db = connect()
    try:
        with db:
            db.execute("UPDATE datasets SET payload=? WHERE id=?", (json.dumps(data), dataset_id))
    finally:
        db.close()
    return evaluate_dataset(data)


@app.get("/templates/transactions")
def transaction_template():
    row = generate_records(1, 77)[0]
    for key in ("timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "geo_country", "asn"):
        row.pop(key, None)
    row["label"] = 0
    return [row]


@app.get("/model-card")
def model_card():
    return {"name":"Isolation Forest", "version":MODEL_VERSION, "features":FEATURES,
            "training":{"count":600,"seed":9182,"synthetic":True},
            "calibration":{"count":300,"seed":3817,"synthetic":True},
            "n_estimators":100,"random_state":42,"threshold_percentile":95,
            "labels_used_for_training":False,"uploaded_data_retrains_model":False,
            "limitations":["No live blockchain ingestion", "No verified identity attribution", "No multi-hop UTXO tracing", "Synthetic reference; real-world accuracy not established"],
            "metrics":{"f1":"Fixed percentile >= 95", "average_precision":"Non-interpolated AP", "pr_auc_trapezoidal":"Trapezoidal area under precision-recall curve using continuous raw scores"}}

# Elliptic training is an explicit local CLI step, never triggered by scoring uploads.
from backend.elliptic import router as elliptic_router
app.include_router(elliptic_router)
