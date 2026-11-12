import base64
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    import app

    with TestClient(app.app) as test_client:
        yield test_client


def pubsub_envelope(transaction, delivery_attempt=1):
    data = base64.b64encode(json.dumps(transaction).encode()).decode()
    return {
        "message": {"data": data, "messageId": "test-message"},
        "subscription": "test-subscription",
        "deliveryAttempt": delivery_attempt,
    }


def sample_transaction():
    return {
        "TX_ID": "pubsub-test",
        "TX_TS": "2026-10-01 09:02:07",
        "CUSTOMER_ID": "7697141589194972",
        "TERMINAL_ID": "73436130",
        "TX_AMOUNT": 22.48,
    }


def test_pubsub_scores_without_mutating_state(client, capsys):
    transaction = sample_transaction()
    predict_request = {
        "transactions": [
            {
                "tx_id": transaction["TX_ID"],
                "tx_ts": transaction["TX_TS"],
                "customer_id": transaction["CUSTOMER_ID"],
                "terminal_id": transaction["TERMINAL_ID"],
                "tx_amount": transaction["TX_AMOUNT"],
            }
        ]
    }
    before = client.post("/predict", json=predict_request).json()["scores"][0]["score"]

    response = client.post("/pubsub", json=pubsub_envelope(transaction, delivery_attempt=2))

    assert response.status_code == 204
    assert response.content == b""
    log = json.loads(capsys.readouterr().out.strip())
    assert log["severity"] == "INFO"
    assert log["message"] == "scored"
    assert log["tx_id"] == transaction["TX_ID"]
    assert log["score"] == pytest.approx(before, abs=1e-6)
    assert log["delivery_attempt"] == 2
    after = client.post("/predict", json=predict_request).json()["scores"][0]["score"]
    assert after == pytest.approx(before, abs=1e-6)


def test_pubsub_rejects_invalid_transaction_and_logs_it(client, capsys):
    transaction = sample_transaction() | {"TX_AMOUNT": -1}

    response = client.post("/pubsub", json=pubsub_envelope(transaction, delivery_attempt=3))

    assert response.status_code == 422
    log = json.loads(capsys.readouterr().out.strip())
    assert log == {
        "severity": "ERROR",
        "message": "rejected",
        "tx_id": transaction["TX_ID"],
        "delivery_attempt": 3,
    }


def test_pubsub_rejects_invalid_base64(client, capsys):
    envelope = {
        "message": {"data": "not-base64!", "messageId": "bad-message"},
        "subscription": "test-subscription",
        "deliveryAttempt": 1,
    }

    response = client.post("/pubsub", json=envelope)

    assert response.status_code == 422
    log = json.loads(capsys.readouterr().out.strip())
    assert log == {
        "severity": "ERROR",
        "message": "rejected",
        "tx_id": None,
        "delivery_attempt": 1,
    }
