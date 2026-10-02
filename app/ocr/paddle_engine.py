"""PaddleOCR 3.x / PP-OCRv6 implementation of the OCREngine interface."""

import logging
import os
from typing import Any, List, Optional
import numpy as np

from app.ocr.engine import EngineResult, ExtractedTextLine, OCREngine

logger = logging.getLogger(__name__)

# Optimize startup and disable PIR / oneDNN attribute conversion bugs in new executor
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"


class PaddleOCREngine(OCREngine):
    """PaddleOCR Engine wrapper targeting PP-OCR pipeline."""

    def __init__(
        self,
        use_angle_cls: bool = True,
        lang: str = "en",
        use_gpu: bool = False,
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.5,
    ):
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self.use_gpu = use_gpu
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh
        self._ocr = None
        self._initialized = False

    def _initialize(self) -> None:
        """Lazy initialization of the PaddleOCR model."""
        if self._initialized:
            return

        try:
            from paddleocr import PaddleOCR

            logger.info("Initializing PaddleOCR engine (lang=%s)...", self.lang)

            # PaddleOCR prefers modern kwargs; fallback cleanly if older version
            try:
                self._ocr = PaddleOCR(
                    use_textline_orientation=self.use_angle_cls,
                    lang=self.lang,
                    text_det_thresh=self.det_db_thresh,
                    text_det_box_thresh=self.det_db_box_thresh,
                    enable_mkldnn=False,
                )
            except (TypeError, ValueError):
                try:
                    self._ocr = PaddleOCR(
                        use_angle_cls=self.use_angle_cls,
                        lang=self.lang,
                        use_gpu=self.use_gpu,
                        det_db_thresh=self.det_db_thresh,
                        det_db_box_thresh=self.det_db_box_thresh,
                        enable_mkldnn=False,
                    )
                except Exception:
                    # Minimal initialization fallback
                    self._ocr = PaddleOCR(lang=self.lang, enable_mkldnn=False)

            self._initialized = True
            logger.info("PaddleOCR engine successfully initialized.")
        except Exception as e:
            logger.error("Failed to initialize PaddleOCR engine: %s", str(e), exc_info=True)
            self._initialized = False
            raise RuntimeError(f"PaddleOCR initialization failed: {e}") from e

    def is_available(self) -> bool:
        """Check if PaddleOCR library is installed and importable."""
        try:
            import paddleocr  # noqa: F401
            import paddle  # noqa: F401
            return True
        except ImportError:
            return False

    def get_name(self) -> str:
        return "paddleocr_ppocrv6"

    def _parse_polygon(self, raw_box: Any) -> tuple[List[List[int]], List[int]]:
        """Parse raw bounding box/polygon points into normalized polygon and axis-aligned bbox."""
        poly: List[List[int]] = []
        x_coords: List[int] = []
        y_coords: List[int] = []

        if isinstance(raw_box, (list, np.ndarray)):
            for pt in raw_box:
                if isinstance(pt, (list, tuple, np.ndarray)) and len(pt) >= 2:
                    x = int(round(float(pt[0])))
                    y = int(round(float(pt[1])))
                    poly.append([x, y])
                    x_coords.append(x)
                    y_coords.append(y)
                elif isinstance(pt, (int, float)):
                    # Axis aligned flat format [x1, y1, x2, y2]
                    pass

        if len(poly) >= 4 and x_coords and y_coords:
            bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        elif isinstance(raw_box, (list, np.ndarray)) and len(raw_box) == 4 and not poly:
            x1, y1, x2, y2 = [int(round(float(v))) for v in raw_box]
            bbox = [x1, y1, x2, y2]
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        else:
            bbox = [0, 0, 0, 0]

        return poly, bbox

    def extract_text(self, image: np.ndarray) -> EngineResult:
        """Run text detection and recognition on the input image using PaddleOCR."""
        self._initialize()

        if image is None or image.size == 0:
            return EngineResult(lines=[], raw_text="", engine_name=self.get_name())

        # Check for uniform blank image to fast-exit
        if np.std(image) < 1.0:
            return EngineResult(lines=[], raw_text="", engine_name=self.get_name(), engine_version="3.x")

        lines: List[ExtractedTextLine] = []
        full_text_parts: List[str] = []

        try:
            # For PaddleOCR 3.x / paddlex pipeline, predict is the primary method
            raw_results = None
            if hasattr(self._ocr, "predict"):
                try:
                    raw_results = self._ocr.predict(image)
                except Exception as pred_err:
                    logger.debug("predict call fallback to ocr: %s", pred_err)
                    raw_results = self._ocr.ocr(image)
            elif hasattr(self._ocr, "ocr"):
                try:
                    raw_results = self._ocr.ocr(image)
                except TypeError:
                    raw_results = self._ocr.ocr(image, cls=self.use_angle_cls)

            # Handle PaddleOCR 3.x / paddlex structure or 2.x structure
            if raw_results is not None:
                # Format A: List of items [[ [box], (text, conf) ], ...]
                if isinstance(raw_results, list) and len(raw_results) > 0:
                    page_items = raw_results[0] if isinstance(raw_results[0], list) else raw_results

                    # Check if page is a dict or object (PaddleOCR 3.x Pipeline Result)
                    if hasattr(raw_results[0], "__getitem__") and isinstance(raw_results[0], dict):
                        # PaddleOCR 3.x dictionary output
                        res_dict = raw_results[0]
                        boxes = res_dict.get("dt_polys") or res_dict.get("dt_boxes") or res_dict.get("boxes", [])
                        texts = res_dict.get("rec_texts") or res_dict.get("texts", [])
                        scores = res_dict.get("rec_scores") or res_dict.get("scores", [])

                        for idx, (box, text, score) in enumerate(zip(boxes, texts, scores)):
                            txt = str(text).strip()
                            if not txt:
                                continue
                            poly, bbox = self._parse_polygon(box)
                            lines.append(
                                ExtractedTextLine(
                                    text=txt,
                                    confidence=round(float(score), 4),
                                    bbox=bbox,
                                    polygon=poly,
                                )
                            )
                            full_text_parts.append(txt)

                    elif page_items is not None and isinstance(page_items, list):
                        # Classic PaddleOCR list of lists format
                        for item in page_items:
                            if not item or not isinstance(item, (list, tuple)) or len(item) < 2:
                                continue

                            box = item[0]
                            text_conf = item[1]

                            if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                                recognized_text = str(text_conf[0]).strip()
                                confidence = float(text_conf[1])
                            elif isinstance(text_conf, str):
                                recognized_text = text_conf.strip()
                                confidence = 1.0
                            else:
                                continue

                            if not recognized_text:
                                continue

                            poly, bbox = self._parse_polygon(box)
                            lines.append(
                                ExtractedTextLine(
                                    text=recognized_text,
                                    confidence=round(confidence, 4),
                                    bbox=bbox,
                                    polygon=poly,
                                )
                            )
                            full_text_parts.append(recognized_text)

        except Exception as e:
            logger.error("PaddleOCR execution error: %s", str(e), exc_info=True)
            raise RuntimeError(f"PaddleOCR processing error: {e}") from e

        combined_text = "\n".join(full_text_parts)

        return EngineResult(
            lines=lines,
            raw_text=combined_text,
            engine_name=self.get_name(),
            engine_version="3.x",
            extra_metadata={
                "line_count": len(lines),
                "lang": self.lang,
            },
        )
