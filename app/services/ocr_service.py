from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import OCRFailedException, NameNotFoundException, AppException, ErrorCode
from app.core.logging import logger
from app.utils.text import extract_candidate_name_from_lines, normalize_text


class OCRService:
    """
    Decoupled Optical Character Recognition (OCR) Service using PaddleOCR.
    Extracts text lines and parses candidate names from identity documents.
    Compatible with both PaddleOCR v2 and v3 response structures.
    """

    def __init__(self, lang: Optional[str] = None, use_angle_cls: Optional[bool] = None):
        self.lang = lang or settings.OCR_LANGUAGE
        self.use_angle_cls = use_angle_cls if use_angle_cls is not None else settings.OCR_USE_ANGLE_CLS
        self._ocr = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the PaddleOCR engine with backward & forward compatibility."""
        if self._initialized:
            return

        try:
            from paddleocr import PaddleOCR

            logger.info(f"Initializing PaddleOCR engine (lang={self.lang})...")
            # Handle both PaddleOCR v3 and v2 constructor signatures
            try:
                self._ocr = PaddleOCR(
                    lang=self.lang,
                    use_angle_cls=self.use_angle_cls,
                )
            except TypeError:
                try:
                    self._ocr = PaddleOCR(
                        lang=self.lang,
                        use_textline_orientation=self.use_angle_cls,
                    )
                except TypeError:
                    self._ocr = PaddleOCR(lang=self.lang)

            self._initialized = True
            logger.info("PaddleOCR engine initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            raise AppException(
                error_code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=f"Failed to initialize OCR engine: {str(e)}",
                status_code=500,
            )

    @property
    def ocr(self):
        if not self._initialized:
            self.initialize()
        return self._ocr

    def extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run OCR on an image and extract all text lines and candidate name.
        Supports both PaddleOCR v2 ([[bbox, (text, conf)]]) and v3 ({'rec_texts': ...}) formats.
        """
        if image is None or image.size == 0:
            raise OCRFailedException("Cannot run OCR on empty or null image.")

        try:
            # PaddleOCR accepts BGR numpy array
            try:
                results = self.ocr.ocr(image, cls=self.use_angle_cls)
            except (TypeError, ValueError):
                results = self.ocr.ocr(image)
        except Exception as e:
            logger.error(f"PaddleOCR execution error: {str(e)}")
            raise OCRFailedException(f"OCR processing failed: {str(e)}")

        lines: List[str] = []
        details: List[Dict[str, Any]] = []

        if results:
            # Flatten if wrapped in single-element outer list
            items = results if isinstance(results, list) else [results]

            for item in items:
                if item is None:
                    continue

                # Format A: PaddleOCR v3 dictionary structure
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

                # Format B: PaddleOCR v2 list of lines structure [[bbox, (text, score)], ...]
                elif isinstance(item, list):
                    for subitem in item:
                        if isinstance(subitem, dict):
                            # Dict inside list
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

        extracted_name = extract_candidate_name_from_lines(lines)

        return {
            "success": len(lines) > 0,
            "lines": lines,
            "details": details,
            "extracted_name": extracted_name,
        }


# Singleton instance
ocr_service = OCRService()
