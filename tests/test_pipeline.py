"""
Integration tests for the Pipeline class.

These tests mock DSPy's ChainOfThought modules to avoid making real API calls
while verifying the Pipeline's orchestration logic.
"""

import pytest
from unittest.mock import Mock, patch
import os

# Mock environment variables before importing app.server
os.environ.setdefault("OPENROUTER_API_KEY", "test-key-for-testing")
os.environ.setdefault("OPENROUTER_MODEL", "test-model")

import dspy
from app.server import Pipeline, Segmenter, Translator


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_prediction():
    """Factory fixture for creating mock Prediction objects."""
    def _create_prediction(**kwargs):
        prediction = Mock(spec=dspy.Prediction)
        for key, value in kwargs.items():
            setattr(prediction, key, value)
        return prediction
    return _create_prediction


@pytest.fixture
def mock_segmenter(mock_prediction):
    """Mock segmenter that returns predefined segments based on input."""
    def _segmenter(text: str):
        # Map input text to expected segments
        segment_map = {
            "你好世界": ["你好", "世界"],
            "我喜欢编程": ["我", "喜欢", "编程"],
            "测试": ["测试"],
            "": [],
        }
        segments = segment_map.get(text, ["默认", "段落"])
        return mock_prediction(segments=segments)

    mock = Mock()
    mock.side_effect = _segmenter
    return mock


@pytest.fixture
def mock_translator(mock_prediction):
    """Mock translator that returns predefined (pinyin, english) tuples for batched segments."""
    def _translator(segments: list[str]):
        # Mapping of Chinese segments to translations
        translation_map = {
            "你好": ("nǐ hǎo", "hello"),
            "世界": ("shì jiè", "world"),
            "我": ("wǒ", "I"),
            "喜欢": ("xǐ huān", "like"),
            "编程": ("biān chéng", "programming"),
            "测试": ("cè shì", "test"),
            "默认": ("mò rèn", "default"),
            "段落": ("duàn luò", "paragraph"),
        }
        results = [translation_map.get(seg, ("unknown", "unknown")) for seg in segments]
        return mock_prediction(results=results)

    mock = Mock()
    mock.side_effect = _translator
    return mock


# ============================================================================
# TEST CASES
# ============================================================================

def test_pipeline_initialization():
    """Verify Pipeline initializes with correct DSPy modules."""
    with patch('dspy.ChainOfThought') as mock_cot, patch('dspy.Predict') as mock_predict:
        mock_cot.return_value = Mock()
        mock_predict.return_value = Mock()

        pipeline = Pipeline()

        # Segmenter uses ChainOfThought
        mock_cot.assert_called_once_with(Segmenter)
        # Translator uses Predict (batched)
        mock_predict.assert_called_once_with(Translator)


def test_pipeline_forward_basic_text(mock_segmenter, mock_translator):
    """Test pipeline with basic Chinese text '你好世界'."""
    with patch('dspy.ChainOfThought') as mock_cot, patch('dspy.Predict') as mock_predict:
        mock_cot.return_value = mock_segmenter
        mock_predict.return_value = mock_translator

        pipeline = Pipeline()
        result = pipeline.forward("你好世界")

        # Should return 2 results (你好, 世界)
        assert len(result) == 2
        assert result[0] == ("你好", "nǐ hǎo", "hello")
        assert result[1] == ("世界", "shì jiè", "world")


def test_pipeline_forward_multiple_segments(mock_segmenter, mock_translator):
    """Test pipeline segments text into multiple words correctly."""
    with patch('dspy.ChainOfThought') as mock_cot, patch('dspy.Predict') as mock_predict:
        mock_cot.return_value = mock_segmenter
        mock_predict.return_value = mock_translator

        pipeline = Pipeline()
        result = pipeline.forward("我喜欢编程")

        assert len(result) == 3
        assert result[0] == ("我", "wǒ", "I")
        assert result[1] == ("喜欢", "xǐ huān", "like")
        assert result[2] == ("编程", "biān chéng", "programming")


def test_pipeline_forward_empty_input(mock_prediction):
    """Test pipeline handles empty input gracefully."""
    with patch('dspy.ChainOfThought') as mock_cot, patch('dspy.Predict') as mock_predict:
        empty_segmenter = Mock(return_value=mock_prediction(segments=[]))
        empty_translator = Mock(return_value=mock_prediction(results=[]))

        mock_cot.return_value = empty_segmenter
        mock_predict.return_value = empty_translator

        pipeline = Pipeline()
        result = pipeline.forward("")

        assert len(result) == 0


def test_pipeline_uses_actual_prediction_objects():
    """Test with actual DSPy Prediction objects."""
    with patch('dspy.ChainOfThought') as mock_cot, patch('dspy.Predict') as mock_predict:
        # Use real Prediction objects
        segment_pred = dspy.Prediction(segments=["你", "好"])
        translation_pred = dspy.Prediction(results=[("nǐ", "you"), ("hǎo", "good")])

        mock_segmenter = Mock(return_value=segment_pred)
        mock_translator = Mock(return_value=translation_pred)

        mock_cot.return_value = mock_segmenter
        mock_predict.return_value = mock_translator

        pipeline = Pipeline()
        result = pipeline.forward("你好")

        # Batched translator called once with all segments
        mock_translator.assert_called_once()
        call_kwargs = mock_translator.call_args[1]
        assert call_kwargs["segments"] == ["你", "好"]

        # Result is correct list of tuples
        assert len(result) == 2
        assert result[0] == ("你", "nǐ", "you")
        assert result[1] == ("好", "hǎo", "good")


def test_pipeline_translator_called_once_with_all_segments(
    mock_segmenter, mock_translator
):
    """Verify translator is called ONCE with all segments (batched)."""
    with patch('dspy.ChainOfThought') as mock_cot, patch('dspy.Predict') as mock_predict:
        mock_cot.return_value = mock_segmenter
        mock_predict.return_value = mock_translator

        pipeline = Pipeline()
        result = pipeline.forward("我喜欢编程")  # 3 segments

        # Translator should be called exactly once (batched)
        assert mock_translator.call_count == 1

        # Verify translator called with all segments at once
        call_kwargs = mock_translator.call_args[1]
        assert call_kwargs["segments"] == ["我", "喜欢", "编程"]

        # Verify results
        assert len(result) == 3
