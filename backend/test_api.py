import csv
import io
import json

import pytest
from fastapi.testclient import TestClient

from backend.main import app, generate_records


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAIN_FORENSICS_DB_PATH", str(tmp_path / "test.sqlite3"))
    with TestClient(app) as session:
        yield session


def test_demo_workflow_and_export(client):
    response = client.post("/demo", json={"count": 30, "seed": 123})
    assert response.status_code == 201
    result = response.json()
    assert result["unique_transactions"] == 30
    assert result["flagged_count"] >= 3
    dataset_id = result["id"]
    alerts = client.get(f"/datasets/{dataset_id}/alerts").json()["alerts"]
    assert alerts and alerts[0]["reasons"] and alerts[0]["source_records"]
    graph = client.get(f"/datasets/{dataset_id}/graph", params={"limit": 2}).json()
    assert len([n for n in graph["nodes"] if n["type"] == "transaction"]) == 2
    assert graph["edges"] and graph["truncated"]
    exported = client.get(f"/datasets/{dataset_id}/export").json()
    assert len(exported["observations"]) == 30
    with TestClient(app) as restarted:
        assert restarted.get(f"/datasets/{dataset_id}").json()["id"] == dataset_id


def test_validation_deduplication_and_relay_observations(client):
    good = generate_records(1, 42)[0]
    second_relay = {**good, "src_ip": "203.0.113.8"}
    invalid = {**good, "output_amounts": [1], "output_addresses": ["x"], "fee": 0}
    conflict = {**good, "input_addresses": ["different"] * len(good["input_addresses"])}
    response = client.post(
        "/datasets",
        files={
            "file": (
                "test.json",
                json.dumps([good, good, second_relay, invalid, conflict]),
                "application/json",
            )
        },
    )
    result = response.json()
    assert response.status_code == 201
    assert result["duplicates"] == 1
    assert result["accepted_observations"] == 2
    assert result["unique_transactions"] == 1
    assert result["rejected_count"] == 2


@pytest.mark.parametrize("extension", ["json", "csv", "xml"])
def test_formats(client, extension):
    rows = generate_records(2, 27)
    if extension == "json":
        text = json.dumps(rows)
    elif extension == "csv":
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value) if isinstance(value, list) else value
                for key, value in row.items()
            }
            for row in rows
        )
        text = stream.getvalue()
    else:
        from xml.etree.ElementTree import Element, SubElement, tostring

        root = Element("records")
        for row in rows:
            element = SubElement(root, "record")
            for key, value in row.items():
                if value is not None:
                    SubElement(element, key).text = (
                        json.dumps(value) if isinstance(value, list) else str(value)
                    )
        text = tostring(root, encoding="unicode")
    response = client.post("/datasets", files={"file": (f"data.{extension}", text)})
    assert response.status_code == 201, response.text
    exported = client.get(f"/datasets/{response.json()['id']}/export").json()
    assert [r["txid"] for r in exported["observations"]] == [r["txid"] for r in rows]
    assert exported["observations"][0]["input_amounts"] == rows[0]["input_amounts"]


def test_input_errors(client):
    assert (
        client.post("/datasets", files={"file": ("bad.json", "{")}).status_code == 422
    )
    assert (
        client.post("/datasets", files={"file": ("bad.json", "[]")}).status_code == 422
    )
    assert (
        client.post("/datasets", files={"file": ("bad.txt", "text")}).status_code == 415
    )
    assert (
        client.post(
            "/datasets", files={"file": ("huge.json", b"x" * (2 * 1024 * 1024 + 1))}
        ).status_code
        == 413
    )
    malicious = '<!DOCTYPE records [<!ENTITY x SYSTEM "file:///etc/passwd">]><records>&x;</records>'
    assert (
        client.post("/datasets", files={"file": ("bad.xml", malicious)}).status_code
        == 422
    )
    assert client.get("/datasets/missing").status_code == 404
    row = generate_records(1, 1)[0]
    row["input_amounts"][0] = 1.5
    assert (
        client.post(
            "/datasets", files={"file": ("fraction.json", json.dumps([row]))}
        ).status_code
        == 422
    )


def test_repeatability(client):
    results = []
    for _ in range(2):
        dataset_id = client.post("/demo", json={"count": 10, "seed": 14}).json()["id"]
        results.append(
            client.get(f"/datasets/{dataset_id}/alerts?flagged_only=false").json()[
                "alerts"
            ]
        )
    assert results[0] == results[1]
