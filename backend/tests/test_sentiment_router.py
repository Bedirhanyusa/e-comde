import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from backend.main import app

client = TestClient(app)

MOCK_RESULTS = [
    {"text": "Ürün tam istediğim gibi, hızlı kargo için teşekkürler.", "label": "olumlu", "confidence": 0.973, "flagged": False},
    {"text": "Beklediğimden çok kötü çıktı, para iadesi talep ettim.", "label": "olumsuz", "confidence": 0.958, "flagged": False},
    {"text": "Ürün harika ancak kargo çok geç geldi.", "label": "notr", "confidence": 0.714, "flagged": True},
]


@patch("backend.routers.sentiment.get_sentiment_classifier")
def test_analyze_returns_results(mock_get):
    mock_clf = MagicMock()
    mock_clf.predict.return_value = MOCK_RESULTS
    mock_get.return_value = mock_clf

    response = client.post(
        "/api/v1/sentiment/analyze",
        json={"texts": [r["text"] for r in MOCK_RESULTS]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert data["flagged_count"] == 1


@patch("backend.routers.sentiment.get_sentiment_classifier")
def test_analyze_empty_list(mock_get):
    mock_clf = MagicMock()
    mock_clf.predict.return_value = []
    mock_get.return_value = mock_clf

    response = client.post("/api/v1/sentiment/analyze", json={"texts": []})
    assert response.status_code == 200
    assert response.json()["total"] == 0


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
