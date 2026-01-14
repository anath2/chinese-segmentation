import io
import os
import re
import unicodedata
from pathlib import Path
from threading import Lock

import dspy
from PIL import Image as PILImage
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
import json
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure DSPy with OpenRouter
openrouter_model = os.getenv("OPENROUTER_MODEL")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is required")

lm = dspy.LM(
    model=f"openrouter/{openrouter_model}",
    api_key=openrouter_api_key,
)
dspy.configure(lm=lm)

# Initialize FastAPI app
app = FastAPI(title="Chinese Text Transcriber", version="1.0.0")

# Configure static files and templates
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Initialize pipeline with thread-safe lazy initialization
_pipeline_lock = Lock()
pipeline = None


def get_pipeline():
    """Thread-safe lazy initialization of pipeline"""
    global pipeline
    if pipeline is None:
        with _pipeline_lock:
            if pipeline is None:
                pipeline = Pipeline()
    return pipeline


# Request/Response Models
class TranslateRequest(BaseModel):
    text: str


class TranslationResult(BaseModel):
    segment: str
    pinyin: str
    english: str


class ParagraphResult(BaseModel):
    translations: list[TranslationResult]
    separator: str


class TranslateResponse(BaseModel):
    paragraphs: list[ParagraphResult]


# Signature Definitions
class Segmenter(dspy.Signature):
    """Segment Chinese text into words"""

    text: str = dspy.InputField(description="Chinese text to segment")
    segments: list[str] = dspy.OutputField(description="List of words")


class Translator(dspy.Signature):
    """Translate Chinese words to Pinyin and English"""

    segment: str = dspy.InputField(description="A segment of chinese text to translate")
    context: str = dspy.InputField(description="Context of the translation task")
    pinyin: str = dspy.OutputField(
        description="Pinyin transliteration of the segment"
    )
    english: str = dspy.OutputField(
        description="English translation of the segment"
    )


class OCRExtractor(dspy.Signature):
    """Extract Chinese text from an image"""

    image: dspy.Image = dspy.InputField(description="Image containing Chinese text")
    chinese_text: str = dspy.OutputField(
        description="Extracted Chinese text from the image"
    )


# Image validation constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def validate_image_file(file_bytes: bytes, filename: str) -> tuple[bool, str | None]:
    """Validate uploaded image file"""
    if len(file_bytes) > MAX_FILE_SIZE:
        return False, "File too large. Maximum size is 5MB"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    return True, None


async def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract Chinese text from image using the configured LM"""
    pil_image = PILImage.open(io.BytesIO(image_bytes))
    original_format = pil_image.format

    # Formats supported by vision APIs - no conversion needed
    supported_formats = {"JPEG", "PNG", "WEBP", "GIF"}

    if original_format in supported_formats:
        # Use original bytes directly if format is supported
        # Only need to handle RGBA/P mode conversion for JPEG
        if original_format == "JPEG" and pil_image.mode in ("RGBA", "P"):
            pil_image = pil_image.convert("RGB")
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            normalized_bytes = buffer.getvalue()
        else:
            normalized_bytes = image_bytes
    else:
        # Convert unsupported formats (MPO, BMP, TIFF, etc.) to JPEG
        if pil_image.mode in ("RGBA", "P"):
            pil_image = pil_image.convert("RGB")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        normalized_bytes = buffer.getvalue()

    image = dspy.Image(normalized_bytes)
    extractor = dspy.ChainOfThought(OCRExtractor)
    result = await extractor.acall(image=image)
    return result.chinese_text


def should_skip_translation(segment: str) -> bool:
    """
    Check if a segment should skip translation.
    Returns True if segment contains only:
    - Whitespace
    - ASCII punctuation/symbols
    - ASCII digits
    - Chinese punctuation
    - Full-width numbers and symbols
    """
    if not segment or not segment.strip():
        return True

    # Define Chinese punctuation marks
    chinese_punctuation = "。，、；：？！""''（）【】《》…—·「」『』〈〉〔〕"

    for char in segment:
        # Skip whitespace
        if char.isspace():
            continue

        # Check if it's ASCII punctuation, symbol, or digit
        if char.isascii() and not char.isalpha():
            continue

        # Check if it's Chinese punctuation
        if char in chinese_punctuation:
            continue

        # Check if it's a full-width number or symbol (Unicode category)
        category = unicodedata.category(char)
        if category in ('Nd', 'No', 'Po', 'Ps', 'Pe', 'Pd', 'Pc', 'Sk', 'Sm', 'So'):
            # Nd: Decimal number, No: Other number
            # Po: Other punctuation, Ps: Open punctuation, Pe: Close punctuation
            # Pd: Dash punctuation, Pc: Connector punctuation
            # Sk: Modifier symbol, Sm: Math symbol, So: Other symbol
            continue

        # If we found a character that's not punctuation/number/symbol, don't skip
        return False

    # All characters are punctuation/numbers/symbols
    return True


def split_into_paragraphs(text: str) -> list[dict[str, str]]:
    """
    Split text into paragraphs while preserving whitespace information.
    Returns a list of dicts with 'content' and 'separator' keys.
    The separator indicates what whitespace follows this paragraph.
    """
    if not text:
        return []

    # Split by newlines while keeping track of the separators
    lines = text.split('\n')
    paragraphs = []

    for i, line in enumerate(lines):
        # Skip completely empty lines at the start
        if not paragraphs and not line.strip():
            continue

        # For non-empty lines, add them as paragraphs
        if line.strip():
            # Determine the separator by looking ahead
            # Count consecutive newlines after this line
            separator = '\n'
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                separator += '\n'
                j += 1

            paragraphs.append({
                'content': line.strip(),
                'separator': separator if i < len(lines) - 1 else ''
            })

    return paragraphs


# Pipeline Definition
class Pipeline(dspy.Module):
    """Pipeline for Chinese text processing"""

    def __init__(self):
        self.segment = dspy.ChainOfThought(Segmenter)
        self.translate = dspy.Predict(Translator)

    def forward(self, text: str) -> list[tuple[str, str, str]]:
        """Sync: Segment and translate Chinese text"""
        segmentation = self.segment(text=text)

        result = []
        for segment in segmentation.segments:
            # Skip translation for segments with only symbols, numbers, and punctuation
            if should_skip_translation(segment):
                result.append((segment, "", ""))
            else:
                translation = self.translate(segment=segment, context=text)
                result.append((segment, translation.pinyin, translation.english))
        return result

    async def aforward(self, text: str) -> list[tuple[str, str, str]]:
        """Async: Segment and translate Chinese text"""
        segmentation = await self.segment.acall(text=text)

        result = []
        for segment in segmentation.segments:
            # Skip translation for segments with only symbols, numbers, and punctuation
            if should_skip_translation(segment):
                result.append((segment, "", ""))
            else:
                translation = await self.translate.acall(segment=segment, context=text)
                result.append((segment, translation.pinyin, translation.english))
        return result


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    """Serve the main page with the translation form"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/translate-text", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest):
    """Translate Chinese text to Pinyin and English"""
    pipe = get_pipeline()

    # Split text into paragraphs
    paragraphs = split_into_paragraphs(request.text)

    # Process each paragraph through the pipeline
    paragraph_results = []
    for para in paragraphs:
        results = await pipe.aforward(para['content'])
        translations = [
            TranslationResult(segment=seg, pinyin=pinyin, english=english)
            for seg, pinyin, english in results
        ]
        paragraph_results.append(
            ParagraphResult(translations=translations, separator=para['separator'])
        )

    return TranslateResponse(paragraphs=paragraph_results)


@app.post("/translate-html", response_class=HTMLResponse)
async def translate_html(request: Request, text: str = Form(...)):
    """Translate Chinese text and return HTML fragment for HTMX."""
    if not text.strip():
        return templates.TemplateResponse(
            "fragments/error.html",
            {"request": request, "message": "Please enter some Chinese text"},
        )

    try:
        pipe = get_pipeline()

        # Split text into paragraphs
        paragraphs = split_into_paragraphs(text)

        # Process each paragraph through the pipeline
        paragraph_results = []
        for para in paragraphs:
            results = await pipe.aforward(para['content'])
            translations = [
                {"segment": seg, "pinyin": pinyin, "english": english}
                for seg, pinyin, english in results
            ]
            paragraph_results.append({
                "translations": translations,
                "separator": para['separator']
            })

        return templates.TemplateResponse(
            "fragments/results.html",
            {"request": request, "paragraphs": paragraph_results, "original_text": text},
        )
    except Exception as e:
        return templates.TemplateResponse(
            "fragments/error.html",
            {"request": request, "message": f"Translation error: {e}"},
        )


@app.post("/translate-stream")
async def translate_stream(text: str = Form(...)):
    """Stream translation progress via SSE"""

    async def generate():
        if not text.strip():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Please enter some Chinese text'})}\n\n"
            return

        try:
            pipe = get_pipeline()

            # Split text into paragraphs
            paragraphs = split_into_paragraphs(text)

            # Count total segments across all paragraphs
            all_paragraph_segments = []
            for para in paragraphs:
                segmentation = await pipe.segment.acall(text=para['content'])
                all_paragraph_segments.append({
                    'segments': segmentation.segments,
                    'separator': para['separator']
                })

            total_segments = sum(len(p['segments']) for p in all_paragraph_segments)

            # Send initial info with paragraph structure
            paragraph_info = [{'segment_count': len(p['segments']), 'separator': p['separator']} for p in all_paragraph_segments]
            yield f"data: {json.dumps({'type': 'start', 'total': total_segments, 'paragraphs': paragraph_info})}\n\n"

            # Step 2: Translate each segment in each paragraph
            global_index = 0
            all_results = []

            for para_idx, para_data in enumerate(all_paragraph_segments):
                para_results = []
                for seg_idx, segment in enumerate(para_data['segments']):
                    # Skip translation for segments with only symbols, numbers, and punctuation
                    if should_skip_translation(segment):
                        result = {
                            "segment": segment,
                            "pinyin": "",
                            "english": "",
                            "index": global_index,
                            "paragraph_index": para_idx,
                        }
                    else:
                        # Use the original paragraph content as context
                        context = paragraphs[para_idx]['content']
                        translation = await pipe.translate.acall(segment=segment, context=context)
                        result = {
                            "segment": segment,
                            "pinyin": translation.pinyin,
                            "english": translation.english,
                            "index": global_index,
                            "paragraph_index": para_idx,
                        }
                    para_results.append(result)
                    global_index += 1

                    # Send progress update
                    yield f"data: {json.dumps({'type': 'progress', 'current': global_index, 'total': total_segments, 'result': result})}\n\n"

                all_results.append({
                    'translations': para_results,
                    'separator': para_data['separator']
                })

            # Send completion with paragraph structure
            yield f"data: {json.dumps({'type': 'complete', 'paragraphs': all_results})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/extract-text-html", response_class=HTMLResponse)
async def extract_text_html(request: Request, file: UploadFile = File(...)):
    """HTMX endpoint for OCR extraction only - fills textarea for editing"""
    try:
        file_bytes = await file.read()

        valid, error = validate_image_file(file_bytes, file.filename or "image.png")
        if not valid:
            return templates.TemplateResponse(
                "fragments/error.html", {"request": request, "message": error}
            )

        extracted_text = await extract_text_from_image(file_bytes)

        if not extracted_text.strip():
            return templates.TemplateResponse(
                "fragments/error.html",
                {"request": request, "message": "No Chinese text found in image"},
            )

        return templates.TemplateResponse(
            "fragments/ocr-result.html",
            {"request": request, "extracted_text": extracted_text},
        )
    except Exception as e:
        return templates.TemplateResponse(
            "fragments/error.html",
            {"request": request, "message": f"OCR error: {e}"},
        )


@app.post("/translate-image", response_model=TranslateResponse)
async def translate_image(file: UploadFile = File(...)):
    """Extract Chinese text from image and translate"""
    file_bytes = await file.read()

    valid, error = validate_image_file(file_bytes, file.filename or "image.png")
    if not valid:
        raise HTTPException(status_code=400, detail=error)

    extracted_text = await extract_text_from_image(file_bytes)

    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="No Chinese text found in image")

    pipe = get_pipeline()

    # Split text into paragraphs
    paragraphs = split_into_paragraphs(extracted_text)

    # Process each paragraph through the pipeline
    paragraph_results = []
    for para in paragraphs:
        results = await pipe.aforward(para['content'])
        translations = [
            TranslationResult(segment=seg, pinyin=pinyin, english=english)
            for seg, pinyin, english in results
        ]
        paragraph_results.append(
            ParagraphResult(translations=translations, separator=para['separator'])
        )

    return TranslateResponse(paragraphs=paragraph_results)


@app.post("/translate-image-html", response_class=HTMLResponse)
async def translate_image_html(request: Request, file: UploadFile = File(...)):
    """HTMX endpoint for image translation"""
    try:
        file_bytes = await file.read()

        valid, error = validate_image_file(file_bytes, file.filename or "image.png")
        if not valid:
            return templates.TemplateResponse(
                "fragments/error.html", {"request": request, "message": error}
            )

        extracted_text = await extract_text_from_image(file_bytes)

        if not extracted_text.strip():
            return templates.TemplateResponse(
                "fragments/error.html",
                {"request": request, "message": "No Chinese text found in image"},
            )

        pipe = get_pipeline()

        # Split text into paragraphs
        paragraphs = split_into_paragraphs(extracted_text)

        # Process each paragraph through the pipeline
        paragraph_results = []
        for para in paragraphs:
            results = await pipe.aforward(para['content'])
            translations = [
                {"segment": seg, "pinyin": pinyin, "english": english}
                for seg, pinyin, english in results
            ]
            paragraph_results.append({
                "translations": translations,
                "separator": para['separator']
            })

        return templates.TemplateResponse(
            "fragments/results.html",
            {"request": request, "paragraphs": paragraph_results, "original_text": extracted_text},
        )
    except Exception as e:
        return templates.TemplateResponse(
            "fragments/error.html",
            {"request": request, "message": f"OCR error: {e}"},
        )
