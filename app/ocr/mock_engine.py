"""Mock / Synthetic OCR Engine for testing, CI/CD, and offline verification."""

from typing import List
import numpy as np

from app.ocr.engine import EngineResult, ExtractedTextLine, OCREngine


class MockOCREngine(OCREngine):
    """Mock OCR engine that returns deterministic text for testing."""

    def __init__(self, default_text: str = "Name: Sunav Sharma\nDocument No: ABC123456"):
        self.default_text = default_text

    def is_available(self) -> bool:
        return True

    def get_name(self) -> str:
        return "mock_engine"

    def extract_text(self, image: np.ndarray) -> EngineResult:
        if image is None or image.size == 0:
            return EngineResult(lines=[], raw_text="", engine_name=self.get_name())

        # Check if image is blank / all uniform
        if np.std(image) < 1.0:
            # Blank image -> return 0 lines
            return EngineResult(lines=[], raw_text="", engine_name=self.get_name())

        lines_raw = [line.strip() for line in self.default_text.split("\n") if line.strip()]
        lines: List[ExtractedTextLine] = []

        h, w = image.shape[:2]
        line_height = max(20, h // max(1, (len(lines_raw) + 2)))

        for i, text in enumerate(lines_raw):
            y_start = 50 + (i * (line_height + 15))
            y_end = min(h - 5, y_start + line_height)
            x_start = 50
            x_end = min(w - 50, x_start + (len(text) * 14))

            bbox = [x_start, y_start, x_end, y_end]
            polygon = [
                [x_start, y_start],
                [x_end, y_start],
                [x_end, y_end],
                [x_start, y_end],
            ]

            lines.append(
                ExtractedTextLine(
                    text=text,
                    confidence=0.96 - (i * 0.02),
                    bbox=bbox,
                    polygon=polygon,
                )
            )

        return EngineResult(
            lines=lines,
            raw_text="\n".join(lines_raw),
            engine_name=self.get_name(),
            engine_version="1.0.0-mock",
        )
