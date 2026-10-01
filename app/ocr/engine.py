"""Abstract OCR Engine Interface and Data Structures.

This module provides the core contract that all OCR engine backends
(PaddleOCR, Mock/Fallback, EasyOCR, Tesseract, etc.) must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class ExtractedTextLine:
    """Represents a single text box recognized by an OCR engine."""

    text: str
    confidence: float
    # [x_min, y_min, x_max, y_max]
    bbox: List[int]
    # 4-point polygon [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    polygon: Optional[List[List[int]]] = None


@dataclass
class EngineResult:
    """Standardized output returned by an OCREngine implementation."""

    lines: List[ExtractedTextLine] = field(default_factory=list)
    raw_text: str = ""
    engine_name: str = "generic"
    engine_version: Optional[str] = None
    extra_metadata: dict = field(default_factory=dict)


class OCREngine(ABC):
    """Abstract Base Class for OCR engines."""

    @abstractmethod
    def extract_text(self, image: np.ndarray) -> EngineResult:
        """Run text detection and recognition on the input image.

        Args:
            image: A valid OpenCV BGR or Grayscale image array (H, W, C) or (H, W).

        Returns:
            EngineResult containing detected lines, text, bounding boxes, and confidence.

        Raises:
            RuntimeError: If OCR processing encounters an unrecoverable failure.
        """
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        """Return the unique identifier/name of this OCR engine."""
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the underlying OCR engine dependencies/weights are ready."""
        raise NotImplementedError
