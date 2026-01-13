# Chinese Text Transcriber

A web application for learning and understanding Chinese text. Paste any Chinese text or upload an image, and get instant word-by-word segmentation with pinyin pronunciation and English translations.

## Features

### Text Processing
- **Word Segmentation** - Intelligently breaks Chinese sentences into individual words and phrases
- **Pinyin with Tones** - Shows pronunciation with proper tone marks (e.g., nǐ hǎo)
- **Contextual Translation** - Translates each word considering the surrounding context for accurate meanings

### Image Support
- **OCR Extraction** - Upload images containing Chinese text (PNG, JPEG, WebP, GIF up to 5MB)
- **Extract Only Mode** - Get just the extracted text without translation
- **Full Pipeline** - Extract and translate in one step

## Getting Started

### Requirements
- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- OpenRouter API key

### Installation

```bash
git clone https://github.com/yourusername/chinese-segmentation.git
cd chinese-segmentation
uv sync
```

### Configuration

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=your_preferred_model
```

### Running

```bash
uv run uvicorn app.server:app --reload
```

Open `http://localhost:8000` in your browser.

## API Reference

### Translate Text

**POST** `/translate-text`

Segment and translate text.

```bash
curl -X POST http://localhost:8000/translate-text \
  -H "Content-Type: application/json" \
  -d '{"text": "我喜欢学习中文"}'
```

```json
{
  "results": [
    {"segment": "我", "pinyin": "wǒ", "english": "I"},
    {"segment": "喜欢", "pinyin": "xǐ huān", "english": "like"},
    {"segment": "学习", "pinyin": "xué xí", "english": "to study"},
    {"segment": "中文", "pinyin": "zhōng wén", "english": "Chinese language"}
  ]
}
```

### Translate Image

**POST** `/translate-image`

Extract text from an image and translate it.

```bash
curl -X POST http://localhost:8000/translate-image \
  -F "file=@image.png"
```

Returns the same response format as `/translate-text`.

### Health Check

**GET** `/health`

Returns server status for monitoring.

## License

MIT
