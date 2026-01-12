"""
Tests for OCR functionality.

Tests the image validation and OCR extraction features.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import os

# Mock environment variables before importing app.server
os.environ.setdefault("OPENROUTER_API_KEY", "test-key-for-testing")
os.environ.setdefault("OPENROUTER_MODEL", "test-model")

from app.server import (
    validate_image_file,
    extract_text_from_image,
    MAX_FILE_SIZE,
)


# ============================================================================
# IMAGE VALIDATION TESTS
# ============================================================================


class TestValidateImageFile:
    """Tests for image file validation."""

    def test_valid_png_file(self):
        """Accept valid PNG file."""
        valid, error = validate_image_file(b"fake image data", "test.png")
        assert valid is True
        assert error is None

    def test_valid_jpg_file(self):
        """Accept valid JPG file."""
        valid, error = validate_image_file(b"fake image data", "test.jpg")
        assert valid is True
        assert error is None

    def test_valid_jpeg_file(self):
        """Accept valid JPEG file."""
        valid, error = validate_image_file(b"fake image data", "test.jpeg")
        assert valid is True
        assert error is None

    def test_valid_webp_file(self):
        """Accept valid WebP file."""
        valid, error = validate_image_file(b"fake image data", "test.webp")
        assert valid is True
        assert error is None

    def test_valid_gif_file(self):
        """Accept valid GIF file."""
        valid, error = validate_image_file(b"fake image data", "test.gif")
        assert valid is True
        assert error is None

    def test_case_insensitive_extension(self):
        """Accept files with uppercase extensions."""
        valid, error = validate_image_file(b"fake image data", "test.PNG")
        assert valid is True
        assert error is None

    def test_invalid_extension_pdf(self):
        """Reject PDF files."""
        valid, error = validate_image_file(b"fake data", "test.pdf")
        assert valid is False
        assert error is not None and "Unsupported file type" in error

    def test_invalid_extension_txt(self):
        """Reject text files."""
        valid, error = validate_image_file(b"fake data", "test.txt")
        assert valid is False
        assert error is not None and "Unsupported file type" in error

    def test_file_too_large(self):
        """Reject files exceeding size limit."""
        large_data = b"x" * (MAX_FILE_SIZE + 1)
        valid, error = validate_image_file(large_data, "test.png")
        assert valid is False
        assert error is not None and "too large" in error

    def test_file_at_size_limit(self):
        """Accept files exactly at size limit."""
        data = b"x" * MAX_FILE_SIZE
        valid, error = validate_image_file(data, "test.png")
        assert valid is True
        assert error is None

    def test_empty_file(self):
        """Accept empty files (validation only checks size and extension)."""
        valid, error = validate_image_file(b"", "test.png")
        assert valid is True
        assert error is None


# ============================================================================
# OCR EXTRACTION TESTS
# ============================================================================


class TestExtractTextFromImage:
    """Tests for OCR text extraction."""

    @patch("app.server.PILImage")
    @patch("dspy.ChainOfThought")
    @patch("dspy.Image")
    def test_extracts_chinese_text(self, mock_image_class, mock_cot, mock_pil):
        """Successfully extract Chinese text from image."""
        # Mock PIL image processing
        mock_pil_image = MagicMock()
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil.open.return_value = mock_pil_image

        mock_result = Mock()
        mock_result.chinese_text = "你好世界"
        mock_extractor = Mock()
        mock_extractor.acall = AsyncMock(return_value=mock_result)
        mock_cot.return_value = mock_extractor

        result = asyncio.run(extract_text_from_image(b"fake image bytes"))

        assert result == "你好世界"
        mock_pil.open.assert_called_once()
        mock_image_class.assert_called_once()

    @patch("app.server.PILImage")
    @patch("dspy.ChainOfThought")
    @patch("dspy.Image")
    def test_returns_empty_string_for_no_text(self, mock_image_class, mock_cot, mock_pil):
        """Return empty string when no text found in image."""
        # Mock PIL image processing
        mock_pil_image = MagicMock()
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil.open.return_value = mock_pil_image

        mock_result = Mock()
        mock_result.chinese_text = ""
        mock_extractor = Mock()
        mock_extractor.acall = AsyncMock(return_value=mock_result)
        mock_cot.return_value = mock_extractor

        result = asyncio.run(extract_text_from_image(b"fake image bytes"))

        assert result == ""

    @patch("app.server.PILImage")
    @patch("dspy.ChainOfThought")
    @patch("dspy.Image")
    def test_handles_multiline_text(self, mock_image_class, mock_cot, mock_pil):
        """Handle multiline Chinese text from image."""
        # Mock PIL image processing
        mock_pil_image = MagicMock()
        mock_pil_image.mode = "RGB"
        mock_pil_image.format = "JPEG"
        mock_pil.open.return_value = mock_pil_image

        mock_result = Mock()
        mock_result.chinese_text = "第一行\n第二行\n第三行"
        mock_extractor = Mock()
        mock_extractor.acall = AsyncMock(return_value=mock_result)
        mock_cot.return_value = mock_extractor

        result = asyncio.run(extract_text_from_image(b"fake image bytes"))

        assert result == "第一行\n第二行\n第三行"
        assert result.count("\n") == 2


# ============================================================================
# ENDPOINT INTEGRATION TESTS
# ============================================================================


class TestTranslateImageEndpoint:
    """Tests for /translate-image endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from app.server import app

        return TestClient(app)

    def test_rejects_invalid_file_type(self, client):
        """Reject non-image files."""
        response = client.post(
            "/translate-image",
            files={"file": ("test.pdf", b"fake data", "application/pdf")},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_rejects_large_files(self, client):
        """Reject files exceeding size limit."""
        large_data = b"x" * (MAX_FILE_SIZE + 1)
        response = client.post(
            "/translate-image",
            files={"file": ("test.png", large_data, "image/png")},
        )
        assert response.status_code == 400
        assert "too large" in response.json()["detail"]

    @patch("app.server.extract_text_from_image", new_callable=AsyncMock)
    @patch("app.server.get_pipeline")
    def test_successful_image_translation(self, mock_pipeline, mock_ocr, client):
        """Successfully process image and return translations."""
        mock_ocr.return_value = "你好"
        mock_pipe = Mock()
        mock_pipe.aforward = AsyncMock(return_value=[("你好", "nǐ hǎo", "hello")])
        mock_pipeline.return_value = mock_pipe

        response = client.post(
            "/translate-image",
            files={"file": ("test.png", b"fake image", "image/png")},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["segment"] == "你好"
        assert data["results"][0]["pinyin"] == "nǐ hǎo"
        assert data["results"][0]["english"] == "hello"

    @patch("app.server.extract_text_from_image")
    def test_no_text_found_in_image(self, mock_ocr, client):
        """Return error when no Chinese text found in image."""
        mock_ocr.return_value = ""

        response = client.post(
            "/translate-image",
            files={"file": ("test.png", b"fake image", "image/png")},
        )

        assert response.status_code == 400
        assert "No Chinese text found" in response.json()["detail"]

    @patch("app.server.extract_text_from_image")
    def test_whitespace_only_text(self, mock_ocr, client):
        """Return error when only whitespace found."""
        mock_ocr.return_value = "   \n\t  "

        response = client.post(
            "/translate-image",
            files={"file": ("test.png", b"fake image", "image/png")},
        )

        assert response.status_code == 400
        assert "No Chinese text found" in response.json()["detail"]


class TestTranslateImageHtmlEndpoint:
    """Tests for /translate-image-html HTMX endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from app.server import app

        return TestClient(app)

    def test_rejects_invalid_file_type_html(self, client):
        """Reject non-image files with HTML error response."""
        response = client.post(
            "/translate-image-html",
            files={"file": ("test.pdf", b"fake data", "application/pdf")},
        )
        assert response.status_code == 200  # HTML endpoint returns 200 with error fragment
        assert "Unsupported file type" in response.text

    @patch("app.server.extract_text_from_image", new_callable=AsyncMock)
    @patch("app.server.get_pipeline")
    def test_successful_image_translation_html(self, mock_pipeline, mock_ocr, client):
        """Successfully process image and return HTML fragment."""
        mock_ocr.return_value = "你好"
        mock_pipe = Mock()
        mock_pipe.aforward = AsyncMock(return_value=[("你好", "nǐ hǎo", "hello")])
        mock_pipeline.return_value = mock_pipe

        response = client.post(
            "/translate-image-html",
            files={"file": ("test.png", b"fake image", "image/png")},
        )

        assert response.status_code == 200
        # Should contain the segment in the response
        assert "你好" in response.text

    @patch("app.server.extract_text_from_image")
    def test_no_text_found_html(self, mock_ocr, client):
        """Return error HTML when no Chinese text found."""
        mock_ocr.return_value = ""

        response = client.post(
            "/translate-image-html",
            files={"file": ("test.png", b"fake image", "image/png")},
        )

        assert response.status_code == 200  # HTML endpoint returns 200 with error fragment
        assert "No Chinese text found" in response.text
