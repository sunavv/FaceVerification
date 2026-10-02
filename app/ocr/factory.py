"""OCR Engine Factory and Registry."""

import logging
from typing import Dict, Type
from app.core.config import settings
from app.ocr.engine import OCREngine
from app.ocr.mock_engine import MockOCREngine

logger = logging.getLogger(__name__)

# Registry of available engine classes
_ENGINE_REGISTRY: Dict[str, Type[OCREngine]] = {
    "mock": MockOCREngine,
}

try:
    from app.ocr.paddle_engine import PaddleOCREngine
    _ENGINE_REGISTRY["paddle"] = PaddleOCREngine
    _ENGINE_REGISTRY["paddleocr"] = PaddleOCREngine
except ImportError:
    logger.warning("PaddleOCR engine not directly importable; only mock engine registered initially.")


def get_ocr_engine(engine_name: str = None) -> OCREngine:
    """Factory function to retrieve an instance of the configured OCR Engine.

    Args:
        engine_name: Optional override of engine name (e.g. 'paddle' or 'mock').

    Returns:
        OCREngine instance.
    """
    target = (engine_name or settings.ocr_engine).lower().strip()

    if target in ("paddle", "paddleocr"):
        try:
            from app.ocr.paddle_engine import PaddleOCREngine
            return PaddleOCREngine(
                use_angle_cls=settings.paddle_use_angle_cls,
                lang=settings.paddle_lang,
                use_gpu=settings.paddle_use_gpu,
                det_db_thresh=settings.paddle_det_db_thresh,
                det_db_box_thresh=settings.paddle_det_db_box_thresh,
            )
        except Exception as e:
            logger.warning("Falling back to MockOCREngine due to PaddleOCR load error: %s", str(e))
            return MockOCREngine()

    if target == "mock":
        return MockOCREngine()

    logger.warning("Unknown engine '%s' requested. Defaulting to mock engine.", target)
    return MockOCREngine()
