from __future__ import annotations
import pytest
from fastapi.testclient import TestClient


def test_get_health(client: TestClient):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_post_batch_telemetry(client: TestClient):
    batch = {
        "readings": [
            {
                "reading_id": "b-01",
                "asset_id": "boiler_01",
                "sensor_id": "temperature_01",
                "timestamp": "2026-10-27T08:00:00Z",
                "sequence_number": 1,
                "value": 71.0,
                "unit": "C",
                "source_type": "simulated",
            },
            {
                "reading_id": "b-02",
                "asset_id": "pump_01",
                "sensor_id": "vibration_01",
                "timestamp": "2026-10-27T08:00:00Z",
                "sequence_number": 1,
                "value": 2.65,
                "unit": "m/s2",
                "source_type": "simulated",
            },
        ]
    }
    res = client.post("/api/telemetry", json=batch)
    assert res.status_code == 201
    data = res.json()
    assert data["total_received"] == 2
    assert data["processed"] == 2
    assert len(data["results"]) == 2


def test_get_readings_filtering(client: TestClient):
    client.post("/api/telemetry", json={
        "reading_id": "filter-b",
        "asset_id": "boiler_01",
        "sensor_id": "temperature_01",
        "timestamp": "2026-10-27T08:00:00Z",
        "sequence_number": 1,
        "value": 74.0,
        "unit": "C",
        "source_type": "simulated",
    })
    client.post("/api/telemetry", json={
        "reading_id": "filter-p",
        "asset_id": "pump_01",
        "sensor_id": "vibration_01",
        "timestamp": "2026-10-27T08:00:00Z",
        "sequence_number": 1,
        "value": 2.9,
        "unit": "m/s2",
        "source_type": "simulated",
    })

    # Filter by asset
    res_b = client.get("/api/readings?asset_id=boiler_01")
    assert res_b.status_code == 200
    assert len(res_b.json()) == 1
    assert res_b.json()[0]["asset_id"] == "boiler_01"

    # Filter by sensor
    res_p = client.get("/api/readings?sensor_id=vibration_01")
    assert res_p.status_code == 200
    assert len(res_p.json()) == 1
    assert res_p.json()[0]["sensor_id"] == "vibration_01"


def test_get_event_by_id_and_404(client: TestClient):
    # Ingest out of range reading to generate event
    res = client.post("/api/telemetry", json={
        "reading_id": "evt-gen-01",
        "asset_id": "boiler_01",
        "sensor_id": "temperature_01",
        "timestamp": "2026-10-27T08:00:00Z",
        "sequence_number": 1,
        "value": 300.0,
        "unit": "C",
        "source_type": "simulated",
    })
    assert res.status_code == 201

    events = client.get("/api/events").json()
    assert len(events) >= 1
    event_id = events[0]["event_id"]

    # Fetch specific event
    single_res = client.get(f"/api/events/{event_id}")
    assert single_res.status_code == 200
    assert single_res.json()["event_id"] == event_id

    # Non-existent event returns 404
    missing_res = client.get("/api/events/EVT-NONEXISTENT")
    assert missing_res.status_code == 404
