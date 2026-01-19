"""Tests for the persistence API endpoints (Milestone 0)."""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """Create test client with temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        monkeypatch.setenv("TRANSCRIBER_DB_PATH", str(db_path))
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("OPENROUTER_MODEL", "test-model")

        from app.server import app

        with TestClient(app) as client:
            yield client


class TestTextsApi:
    def test_create_text_success(self, client):
        """POST /api/texts creates a text and returns id."""
        response = client.post(
            "/api/texts",
            json={"raw_text": "你好世界", "source_type": "text", "metadata": {}},
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert len(data["id"]) == 32

    def test_create_text_empty_rejected(self, client):
        """POST /api/texts rejects empty raw_text."""
        response = client.post(
            "/api/texts",
            json={"raw_text": "   ", "source_type": "text"},
        )
        assert response.status_code == 400
        assert "raw_text" in response.json()["detail"].lower()

    def test_get_text_success(self, client):
        """GET /api/texts/{id} retrieves a created text."""
        create_resp = client.post(
            "/api/texts",
            json={"raw_text": "测试文本", "source_type": "ocr", "metadata": {"key": "val"}},
        )
        text_id = create_resp.json()["id"]

        get_resp = client.get(f"/api/texts/{text_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["id"] == text_id
        assert data["raw_text"] == "测试文本"
        assert data["source_type"] == "ocr"
        assert data["metadata"] == {"key": "val"}

    def test_get_text_not_found(self, client):
        """GET /api/texts/{id} returns 404 for missing id."""
        response = client.get("/api/texts/nonexistent_id_12345")
        assert response.status_code == 404


class TestEventsApi:
    def test_create_event_success(self, client):
        """POST /api/events creates an event and returns id."""
        response = client.post(
            "/api/events",
            json={"event_type": "tap", "payload": {"headword": "你好"}},
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert len(data["id"]) == 32

    def test_create_event_with_text_reference(self, client):
        """POST /api/events can reference a text_id."""
        text_resp = client.post(
            "/api/texts",
            json={"raw_text": "测试", "source_type": "text"},
        )
        text_id = text_resp.json()["id"]

        event_resp = client.post(
            "/api/events",
            json={"event_type": "view", "text_id": text_id, "payload": {}},
        )
        assert event_resp.status_code == 200

    def test_create_event_empty_type_rejected(self, client):
        """POST /api/events rejects empty event_type."""
        response = client.post(
            "/api/events",
            json={"event_type": "   ", "payload": {}},
        )
        assert response.status_code == 400


class TestVocabApi:
    def test_save_vocab_success(self, client):
        """POST /api/vocab/save creates a vocab item."""
        response = client.post(
            "/api/vocab/save",
            json={
                "headword": "学习",
                "pinyin": "xué xí",
                "english": "to study",
                "snippet": "我喜欢学习",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "vocab_item_id" in data
        assert len(data["vocab_item_id"]) == 32

    def test_save_vocab_empty_headword_rejected(self, client):
        """POST /api/vocab/save rejects empty headword."""
        response = client.post(
            "/api/vocab/save",
            json={"headword": "   ", "pinyin": "test", "english": "test"},
        )
        assert response.status_code == 400

    def test_save_vocab_upsert_returns_same_id(self, client):
        """POST /api/vocab/save returns same id for duplicate."""
        payload = {
            "headword": "重复",
            "pinyin": "chóng fù",
            "english": "repeat",
        }
        resp1 = client.post("/api/vocab/save", json=payload)
        resp2 = client.post("/api/vocab/save", json=payload)

        assert resp1.json()["vocab_item_id"] == resp2.json()["vocab_item_id"]

    def test_update_vocab_status_success(self, client):
        """POST /api/vocab/status updates status."""
        # First create a vocab item
        save_resp = client.post(
            "/api/vocab/save",
            json={"headword": "状态", "pinyin": "zhuàng tài", "english": "status"},
        )
        vocab_id = save_resp.json()["vocab_item_id"]

        # Update status
        status_resp = client.post(
            "/api/vocab/status",
            json={"vocab_item_id": vocab_id, "status": "learning"},
        )
        assert status_resp.status_code == 200
        assert status_resp.json()["ok"] is True

    def test_update_vocab_status_invalid_rejected(self, client):
        """POST /api/vocab/status rejects invalid status."""
        save_resp = client.post(
            "/api/vocab/save",
            json={"headword": "错误", "pinyin": "cuò wù", "english": "error"},
        )
        vocab_id = save_resp.json()["vocab_item_id"]

        status_resp = client.post(
            "/api/vocab/status",
            json={"vocab_item_id": vocab_id, "status": "invalid"},
        )
        assert status_resp.status_code == 400
