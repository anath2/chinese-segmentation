"""Tests for the persistence layer (Milestone 0)."""

import tempfile
from pathlib import Path

import pytest

from app.persistence import (
    create_event,
    create_text,
    get_db_path,
    get_text,
    init_db,
    save_vocab_item,
    update_vocab_status,
)


@pytest.fixture
def temp_db(monkeypatch):
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        monkeypatch.setenv("TRANSCRIBER_DB_PATH", str(db_path))
        init_db()
        yield db_path


class TestGetDbPath:
    def test_default_path(self, monkeypatch):
        """Uses default path when env var not set."""
        monkeypatch.delenv("TRANSCRIBER_DB_PATH", raising=False)
        path = get_db_path()
        assert path.name == "transcriber.db"
        assert "data" in str(path)

    def test_custom_path_from_env(self, monkeypatch):
        """Uses path from environment variable."""
        monkeypatch.setenv("TRANSCRIBER_DB_PATH", "/tmp/custom.db")
        path = get_db_path()
        # On macOS, /tmp resolves to /private/tmp
        assert path.name == "custom.db"
        assert "tmp" in str(path)


class TestInitDb:
    def test_creates_tables(self, temp_db):
        """init_db creates all required tables."""
        import sqlite3

        conn = sqlite3.connect(str(temp_db))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        expected_tables = {
            "schema_migrations",
            "texts",
            "segments",
            "events",
            "vocab_items",
            "vocab_occurrences",
            "srs_state",
        }
        assert expected_tables.issubset(tables)

    def test_migration_is_idempotent(self, temp_db):
        """Running init_db multiple times is safe."""
        # Already initialized in fixture, run again
        init_db()
        init_db()
        # Should not raise


class TestTextCrud:
    def test_create_text_returns_record(self, temp_db):
        """create_text returns a TextRecord with generated id."""
        record = create_text(
            raw_text="你好世界",
            source_type="text",
            metadata={"key": "value"},
        )
        assert record.id is not None
        assert len(record.id) == 32  # UUID hex
        assert record.raw_text == "你好世界"
        assert record.source_type == "text"
        assert record.metadata == {"key": "value"}
        assert record.created_at is not None

    def test_create_text_normalizes_whitespace(self, temp_db):
        """create_text strips leading/trailing whitespace."""
        record = create_text(
            raw_text="  你好  ",
            source_type="text",
            metadata=None,
        )
        assert record.normalized_text == "你好"

    def test_get_text_returns_record(self, temp_db):
        """get_text retrieves a previously created text."""
        created = create_text(
            raw_text="测试文本",
            source_type="ocr",
            metadata={"source": "image.png"},
        )
        retrieved = get_text(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.raw_text == "测试文本"
        assert retrieved.source_type == "ocr"
        assert retrieved.metadata == {"source": "image.png"}

    def test_get_text_returns_none_for_missing(self, temp_db):
        """get_text returns None for non-existent id."""
        result = get_text("nonexistent_id")
        assert result is None


class TestEventCrud:
    def test_create_event_returns_id(self, temp_db):
        """create_event returns the event id."""
        event_id = create_event(
            event_type="tap",
            text_id=None,
            segment_id=None,
            payload={"headword": "你好"},
        )
        assert event_id is not None
        assert len(event_id) == 32

    def test_create_event_with_text_reference(self, temp_db):
        """create_event can reference a text."""
        text = create_text(raw_text="测试", source_type="text", metadata=None)
        event_id = create_event(
            event_type="view",
            text_id=text.id,
            segment_id=None,
            payload={},
        )
        assert event_id is not None


class TestVocabCrud:
    def test_save_vocab_item_creates_new(self, temp_db):
        """save_vocab_item creates a new vocab item."""
        vocab_id = save_vocab_item(
            headword="学习",
            pinyin="xué xí",
            english="to study",
            text_id=None,
            segment_id=None,
            snippet="我喜欢学习中文",
        )
        assert vocab_id is not None
        assert len(vocab_id) == 32

    def test_save_vocab_item_upserts_on_duplicate(self, temp_db):
        """save_vocab_item returns existing id for duplicate headword/pinyin/english."""
        vocab_id_1 = save_vocab_item(
            headword="学习",
            pinyin="xué xí",
            english="to study",
            text_id=None,
            segment_id=None,
            snippet="snippet 1",
        )
        vocab_id_2 = save_vocab_item(
            headword="学习",
            pinyin="xué xí",
            english="to study",
            text_id=None,
            segment_id=None,
            snippet="snippet 2",
        )
        # Same vocab item
        assert vocab_id_1 == vocab_id_2

    def test_save_vocab_item_different_senses_are_distinct(self, temp_db):
        """Different pinyin/english for same headword creates new items."""
        vocab_id_1 = save_vocab_item(
            headword="行",
            pinyin="xíng",
            english="to walk",
            text_id=None,
            segment_id=None,
            snippet=None,
        )
        vocab_id_2 = save_vocab_item(
            headword="行",
            pinyin="háng",
            english="row, line",
            text_id=None,
            segment_id=None,
            snippet=None,
        )
        assert vocab_id_1 != vocab_id_2

    def test_update_vocab_status(self, temp_db):
        """update_vocab_status changes the status."""
        vocab_id = save_vocab_item(
            headword="测试",
            pinyin="cè shì",
            english="test",
            text_id=None,
            segment_id=None,
            snippet=None,
        )
        # Should not raise
        update_vocab_status(vocab_item_id=vocab_id, status="learning")
        update_vocab_status(vocab_item_id=vocab_id, status="known")

    def test_update_vocab_status_rejects_invalid(self, temp_db):
        """update_vocab_status raises for invalid status."""
        vocab_id = save_vocab_item(
            headword="错误",
            pinyin="cuò wù",
            english="error",
            text_id=None,
            segment_id=None,
            snippet=None,
        )
        with pytest.raises(ValueError, match="Invalid status"):
            update_vocab_status(vocab_item_id=vocab_id, status="invalid_status")
