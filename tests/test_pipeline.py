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
from app.server import Pipeline, Segmenter, Translator, should_skip_translation


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
    """Mock translator that returns pinyin and english for a single segment."""
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
        "你": ("nǐ", "you"),
        "好": ("hǎo", "good"),
    }

    def _translator(segment: str, context: str):
        pinyin, english = translation_map.get(segment, ("unknown", "unknown"))
        return mock_prediction(pinyin=pinyin, english=english)

    mock = Mock()
    mock.side_effect = _translator
    return mock


# ============================================================================
# TEST CASES
# ============================================================================

def test_should_skip_translation():
    """Test the should_skip_translation helper function."""
    # Should skip: empty string and whitespace
    assert should_skip_translation("") is True
    assert should_skip_translation("   ") is True
    assert should_skip_translation("\n\t") is True

    # Should skip: ASCII punctuation and symbols
    assert should_skip_translation(",") is True
    assert should_skip_translation("!") is True
    assert should_skip_translation("?") is True
    assert should_skip_translation(".") is True
    assert should_skip_translation("...") is True
    assert should_skip_translation("@#$%") is True

    # Should skip: ASCII numbers
    assert should_skip_translation("123") is True
    assert should_skip_translation("0") is True
    assert should_skip_translation("3.14") is True

    # Should skip: Chinese punctuation
    assert should_skip_translation("，") is True  # Chinese comma
    assert should_skip_translation("。") is True  # Chinese period
    assert should_skip_translation("！") is True  # Chinese exclamation
    assert should_skip_translation("？") is True  # Chinese question mark
    assert should_skip_translation("、") is True  # Chinese enumeration comma
    assert should_skip_translation("；") is True  # Chinese semicolon
    assert should_skip_translation("：") is True  # Chinese colon
    assert should_skip_translation('"') is True  # Chinese quote
    assert should_skip_translation("'") is True  # Chinese quote
    assert should_skip_translation("（") is True  # Chinese left paren
    assert should_skip_translation("）") is True  # Chinese right paren
    assert should_skip_translation("【") is True  # Chinese bracket
    assert should_skip_translation("】") is True  # Chinese bracket

    # Should skip: mixed punctuation and numbers
    assert should_skip_translation("123,456") is True
    assert should_skip_translation("...!!!") is True
    assert should_skip_translation("，。、") is True

    # Should NOT skip: Chinese characters
    assert should_skip_translation("你好") is False
    assert should_skip_translation("世界") is False
    assert should_skip_translation("我") is False

    # Should NOT skip: mixed Chinese and punctuation (contains Chinese)
    assert should_skip_translation("你好，") is False
    assert should_skip_translation("，你好") is False
    assert should_skip_translation("123你好") is False

    # Should NOT skip: ASCII letters
    assert should_skip_translation("hello") is False
    assert should_skip_translation("a") is False


def test_pipeline_initialization():
    """Verify Pipeline initializes with correct DSPy modules."""
    with patch('dspy.ChainOfThought') as mock_cot, patch('dspy.Predict') as mock_predict:
        mock_cot.return_value = Mock()
        mock_predict.return_value = Mock()

        pipeline = Pipeline()

        # Segmenter uses ChainOfThought
        mock_cot.assert_called_once_with(Segmenter)
        # Translator uses Predict
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
        empty_translator = Mock(return_value=mock_prediction(pinyin="", english=""))

        mock_cot.return_value = empty_segmenter
        mock_predict.return_value = empty_translator

        pipeline = Pipeline()
        result = pipeline.forward("")

        assert len(result) == 0


def test_pipeline_uses_actual_prediction_objects(mock_prediction):
    """Test with actual DSPy Prediction objects."""
    with patch('dspy.ChainOfThought') as mock_cot, patch('dspy.Predict') as mock_predict:
        # Use real Prediction objects
        segment_pred = dspy.Prediction(segments=["你", "好"])

        mock_segmenter = Mock(return_value=segment_pred)

        # Mock translator to return different values for each segment
        def translator_side_effect(segment, context):
            if segment == "你":
                return mock_prediction(pinyin="nǐ", english="you")
            else:
                return mock_prediction(pinyin="hǎo", english="good")

        mock_translator = Mock(side_effect=translator_side_effect)

        mock_cot.return_value = mock_segmenter
        mock_predict.return_value = mock_translator

        pipeline = Pipeline()
        result = pipeline.forward("你好")

        # Translator called once per segment
        assert mock_translator.call_count == 2

        # Result is correct list of tuples
        assert len(result) == 2
        assert result[0] == ("你", "nǐ", "you")
        assert result[1] == ("好", "hǎo", "good")


def test_pipeline_translator_called_per_segment(mock_segmenter, mock_translator):
    """Verify translator is called once per segment (not batched)."""
    with patch('dspy.ChainOfThought') as mock_cot, patch('dspy.Predict') as mock_predict:
        mock_cot.return_value = mock_segmenter
        mock_predict.return_value = mock_translator

        pipeline = Pipeline()
        result = pipeline.forward("我喜欢编程")  # 3 segments

        # Translator should be called once per segment
        assert mock_translator.call_count == 3

        # Verify translator called with each segment and context
        calls = mock_translator.call_args_list
        assert calls[0][1]["segment"] == "我"
        assert calls[0][1]["context"] == "我喜欢编程"
        assert calls[1][1]["segment"] == "喜欢"
        assert calls[1][1]["context"] == "我喜欢编程"
        assert calls[2][1]["segment"] == "编程"
        assert calls[2][1]["context"] == "我喜欢编程"

        # Verify results
        assert len(result) == 3


def test_pipeline_skips_punctuation_and_symbols(mock_prediction):
    """Verify pipeline skips translation for symbols, numbers, and punctuation."""
    with patch('dspy.ChainOfThought') as mock_cot, patch('dspy.Predict') as mock_predict:
        # Create a segmenter that returns mixed segments (Chinese words and punctuation)
        def custom_segmenter(text):
            # Simulate segmentation of: "你好，世界！123"
            # Should segment to: ["你好", "，", "世界", "！", "123"]
            return mock_prediction(segments=["你好", "，", "世界", "！", "123"])

        mock_seg = Mock(side_effect=custom_segmenter)

        # Mock translator - should only be called for actual Chinese words
        def custom_translator(segment, context):
            if segment == "你好":
                return mock_prediction(pinyin="nǐ hǎo", english="hello")
            elif segment == "世界":
                return mock_prediction(pinyin="shì jiè", english="world")
            else:
                return mock_prediction(pinyin="", english="")

        mock_trans = Mock(side_effect=custom_translator)

        mock_cot.return_value = mock_seg
        mock_predict.return_value = mock_trans

        pipeline = Pipeline()
        result = pipeline.forward("你好，世界！123")

        # Should return all 5 segments
        assert len(result) == 5

        # Chinese words should have translations
        assert result[0] == ("你好", "nǐ hǎo", "hello")
        assert result[2] == ("世界", "shì jiè", "world")

        # Punctuation and numbers should have empty translations
        assert result[1] == ("，", "", "")  # Chinese comma
        assert result[3] == ("！", "", "")  # Chinese exclamation
        assert result[4] == ("123", "", "")  # Numbers

        # Translator should only be called twice (for Chinese words, not for punctuation/numbers)
        assert mock_trans.call_count == 2
