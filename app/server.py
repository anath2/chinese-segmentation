import io
import os
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


class TranslateResponse(BaseModel):
    results: list[TranslationResult]


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
            translation = self.translate(segment=segment, context=text)
            result.append((segment, translation.pinyin, translation.english))
        return result

    async def aforward(self, text: str) -> list[tuple[str, str, str]]:
        """Async: Segment and translate Chinese text"""
        segmentation = await self.segment.acall(text=text)

        result = []
        for segment in segmentation.segments:
            translation = await self.translate.acall(segment=segment, context=text)
            result.append((segment, translation.pinyin, translation.english))
        return result


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    """Serve the main page with the translation form"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/translate-text", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest):
    """Translate Chinese text to Pinyin and English"""
    pipe = get_pipeline()

    # Process the text through the pipeline (async)
    results = await pipe.acall(request.text)

    # Format the response
    translations = [
        TranslationResult(segment=seg, pinyin=pinyin, english=english)
        for seg, pinyin, english in results
    ]

    return TranslateResponse(results=translations)


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
        results = await pipe.aforward(text)

        translations = [
            {"segment": seg, "pinyin": pinyin, "english": english}
            for seg, pinyin, english in results
        ]

        return templates.TemplateResponse(
            "fragments/results.html",
            {"request": request, "results": translations, "original_text": text},
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

            # Step 1: Segment
            segmentation = await pipe.segment.acall(text=text)
            segments = segmentation.segments
            total = len(segments)

            # Send segment count and all segments
            yield f"data: {json.dumps({'type': 'start', 'total': total, 'segments': segments})}\n\n"

            # Step 2: Translate each segment
            results = []
            for i, segment in enumerate(segments):
                translation = await pipe.translate.acall(segment=segment, context=text)
                result = {
                    "segment": segment,
                    "pinyin": translation.pinyin,
                    "english": translation.english,
                    "index": i,
                }
                results.append(result)

                # Send progress update
                yield f"data: {json.dumps({'type': 'progress', 'current': i + 1, 'total': total, 'result': result})}\n\n"

            # Send completion
            yield f"data: {json.dumps({'type': 'complete', 'results': results})}\n\n"

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
    results = await pipe.aforward(extracted_text)

    translations = [
        TranslationResult(segment=seg, pinyin=pinyin, english=english)
        for seg, pinyin, english in results
    ]

    return TranslateResponse(results=translations)


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
        results = await pipe.aforward(extracted_text)

        translations = [
            {"segment": seg, "pinyin": pinyin, "english": english}
            for seg, pinyin, english in results
        ]

        return templates.TemplateResponse(
            "fragments/results.html",
            {"request": request, "results": translations, "original_text": extracted_text},
        )
    except Exception as e:
        return templates.TemplateResponse(
            "fragments/error.html",
            {"request": request, "message": f"OCR error: {e}"},
        )
