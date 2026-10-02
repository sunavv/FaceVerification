"""OCR Service Layer.

Orchestrates document validation, preprocessing, OCR engine execution,
text normalization, and response assembly. Provides both high-level document
processing for the OCR API and lightweight text/name extraction for verification.
"""

import logging
import mimetypes
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np

from app.core.config import settings
from app.core.exceptions import OCRFailedException
from app.models.response import DocumentMetadata, OCRResponse, PreprocessingInfo, TextLine
from app.ocr.engine import OCREngine
from app.ocr.factory import get_ocr_engine
from app.ocr.preprocessing import ImagePreprocessor
from app.utils.text import clean_line_text, normalize_text, extract_candidate_name_from_lines

logger = logging.getLogger(__name__)


class OCRService:
    """Service handling high-level OCR workflows and verification text extraction."""

    def __init__(
        self,
        engine: Optional[OCREngine] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
    ):
        self.engine = engine or get_ocr_engine()
        self.preprocessor = preprocessor or ImagePreprocessor(
            enable_deskew=settings.deskew_enabled,
            enable_clahe=settings.clahe_enabled,
            enable_denoise=settings.denoise_enabled,
        )
        self._ocr = None
        self._initialized = False

    def validate_document(self, filename: str, content: bytes) -> None:
        """Validate uploaded file before OCR processing.

        Raises:
            ValueError: If file fails validation criteria.
        """
        if not content or len(content) == 0:
            raise ValueError("Document file is empty (0 bytes).")

        max_bytes = settings.max_image_size_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise ValueError(f"Document exceeds maximum allowable size of {settings.max_image_size_mb}MB.")

        # Validate extension
        _, ext = os.path.splitext(filename)
        clean_ext = ext.lstrip(".").lower()
        if clean_ext not in settings.allowed_extensions:
            raise ValueError(
                f"File format '{ext}' is not supported. Supported extensions: {', '.join(settings.allowed_extensions)}"
            )

    def extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text lines and candidate name from an image.

        Used by VerificationService for 1:1 identity verification.
        Compatible with both direct PaddleOCR instances (or mocks) and OCREngine wrappers.
        """
        if image is None or image.size == 0:
            raise OCRFailedException("Cannot run OCR on empty or null image.")

        lines: List[str] = []
        details: List[Dict[str, Any]] = []

        # 1. Direct PaddleOCR instance (e.g. injected or mocked in unit tests)
        if self._ocr is not None:
            try:
                try:
                    results = self._ocr.ocr(image, cls=True)
                except (TypeError, ValueError):
                    results = self._ocr.ocr(image)
            except Exception as e:
                logger.error(f"PaddleOCR execution error: {str(e)}")
                raise OCRFailedException(f"OCR processing failed: {str(e)}")

            if results:
                items = results if isinstance(results, list) else [results]
                for item in items:
                    if item is None:
                        continue
                    if isinstance(item, dict):
                        rec_texts = item.get("rec_texts", [])
                        rec_scores = item.get("rec_scores", [])
                        rec_polys = item.get("rec_polys", [])
                        for i, text in enumerate(rec_texts):
                            clean_t = str(text).strip()
                            if clean_t:
                                score = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                                poly = rec_polys[i] if i < len(rec_polys) else []
                                lines.append(clean_t)
                                details.append({
                                    "text": clean_t,
                                    "confidence": round(score, 4),
                                    "bbox": poly if isinstance(poly, list) else getattr(poly, "tolist", lambda: [])(),
                                })
                    elif isinstance(item, list):
                        for subitem in item:
                            if isinstance(subitem, dict):
                                clean_t = str(subitem.get("text", "")).strip()
                                if clean_t:
                                    lines.append(clean_t)
                                    details.append({
                                        "text": clean_t,
                                        "confidence": round(float(subitem.get("confidence", 1.0)), 4),
                                        "bbox": subitem.get("bbox", []),
                                    })
                            elif isinstance(subitem, (list, tuple)) and len(subitem) >= 2:
                                bbox = subitem[0]
                                text_tuple = subitem[1]
                                if isinstance(text_tuple, (list, tuple)) and len(text_tuple) >= 2:
                                    text = str(text_tuple[0]).strip()
                                    confidence = float(text_tuple[1])
                                    if text:
                                        lines.append(text)
                                        details.append({
                                            "text": text,
                                            "confidence": round(confidence, 4),
                                            "bbox": bbox,
                                        })
                                elif isinstance(text_tuple, str):
                                    text = text_tuple.strip()
                                    if text:
                                        lines.append(text)
                                        details.append({
                                            "text": text,
                                            "confidence": 1.0,
                                            "bbox": bbox,
                                        })
        else:
            # 2. Use the configured OCREngine
            try:
                engine_result = self.engine.extract_text(image)
                for line in engine_result.lines:
                    cleaned = clean_line_text(line.text)
                    if cleaned:
                        lines.append(cleaned)
                        details.append({
                            "text": cleaned,
                            "confidence": round(float(line.confidence), 4),
                            "bbox": line.bbox,
                        })
            except Exception as e:
                logger.error(f"OCREngine extraction error: {str(e)}")
                raise OCRFailedException(f"OCR processing failed: {str(e)}")

        extracted_name = extract_candidate_name_from_lines(lines)

        return {
            "success": len(lines) > 0,
            "lines": lines,
            "details": details,
            "extracted_name": extracted_name,
        }

    def process_document(self, filename: str, content: bytes) -> OCRResponse:
        """Execute the end-to-end OCR pipeline on an uploaded document."""
        start_time = time.perf_counter()

        # 1. Validation
        self.validate_document(filename, content)

        # 2. Image Decoding
        image = self.preprocessor.decode_image_bytes(content)
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1

        mime_type, _ = mimetypes.guess_type(filename)
        doc_meta = DocumentMetadata(
            filename=filename,
            type=mime_type or "application/octet-stream",
            width=w,
            height=h,
            channels=channels,
            size_bytes=len(content),
        )

        # 3. Preprocessing
        prep_info: Optional[PreprocessingInfo] = None
        if settings.enable_preprocessing:
            prep_res = self.preprocessor.process(image)
            image_to_ocr = prep_res.processed_image
            prep_info = PreprocessingInfo(
                applied=True,
                deskew_angle=prep_res.deskew_angle,
                clahe_applied=prep_res.clahe_applied,
                denoise_applied=prep_res.denoise_applied,
            )
        else:
            image_to_ocr = image
            prep_info = PreprocessingInfo(applied=False)

        # 4. OCR Extraction
        engine_result = self.engine.extract_text(image_to_ocr)

        # 5. Text Normalization and Structuring
        structured_lines: list[TextLine] = []
        for line in engine_result.lines:
            cleaned = clean_line_text(line.text)
            if cleaned:
                structured_lines.append(
                    TextLine(
                        text=cleaned,
                        confidence=line.confidence,
                        bbox=line.bbox,
                        polygon=line.polygon,
                    )
                )

        normalized_full_text = normalize_text(
            "\n".join(l.text for l in structured_lines) if structured_lines else engine_result.raw_text
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return OCRResponse(
            success=True,
            processing_time_ms=elapsed_ms,
            document=doc_meta,
            text=normalized_full_text,
            lines=structured_lines,
            preprocessing=prep_info,
            engine=engine_result.engine_name,
            error=None,
        )


# Singleton instance
ocr_service = OCRService()
