"""Image preprocessing pipeline using OpenCV.

Provides resilient decoding, automatic deskewing, CLAHE contrast enhancement,
and denoising to improve OCR accuracy on scanned/camera documents.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class PreprocessResult:
    """Result of image preprocessing."""

    processed_image: np.ndarray
    deskew_angle: Optional[float] = None
    clahe_applied: bool = False
    denoise_applied: bool = False


class ImagePreprocessor:
    """OpenCV-based image preprocessing module."""

    def __init__(
        self,
        enable_deskew: bool = True,
        enable_clahe: bool = True,
        enable_denoise: bool = False,
    ):
        self.enable_deskew = enable_deskew
        self.enable_clahe = enable_clahe
        self.enable_denoise = enable_denoise

    @staticmethod
    def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
        """Decode raw image bytes into an OpenCV BGR numpy array.

        Args:
            image_bytes: Raw binary file bytes.

        Returns:
            np.ndarray: OpenCV BGR image array.

        Raises:
            ValueError: If bytes cannot be decoded into a valid image.
        """
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Empty image data provided")

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image. Unsupported or corrupted file format.")

        return img

    def compute_skew_angle(self, gray_image: np.ndarray) -> float:
        """Compute document skew angle in degrees using thresholding and minAreaRect."""
        # Threshold the image (invert so text is white on black background)
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Dilate text regions to connect characters into solid lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        # Find non-zero points
        points = cv2.findNonZero(dilated)
        if points is None or len(points) < 10:
            return 0.0

        # Compute minimum area bounding box
        rect = cv2.minAreaRect(points)
        angle = rect[-1]

        # Adjust angle according to OpenCV minAreaRect conventions
        if angle < -45.0:
            angle = -(90.0 + angle)
        elif angle > 45.0:
            angle = -(90.0 - angle)
        else:
            angle = -angle

        # If angle is negligible, ignore
        if abs(angle) < 0.3 or abs(angle) > 45.0:
            return 0.0

        return float(angle)

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by the given angle around its center with border replication."""
        if abs(angle) < 0.01:
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding dimensions to avoid clipping rotated content
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust matrix for translation
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization to enhance text clarity."""
        if len(image.shape) == 2:
            # Grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

        # Color image (BGR) -> convert to LAB and apply CLAHE to L-channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def apply_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply gentle bilateral/Gaussian denoising."""
        if len(image.shape) == 2:
            return cv2.bilateralFilter(image, 5, 50, 50)
        return cv2.bilateralFilter(image, 5, 50, 50)

    def process(self, image: np.ndarray) -> PreprocessResult:
        """Execute full preprocessing pipeline on the input image."""
        img = image.copy()
        applied_deskew_angle: Optional[float] = None
        clahe_applied = False
        denoise_applied = False

        # 1. Deskew
        if self.enable_deskew:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            angle = self.compute_skew_angle(gray)
            if abs(angle) >= 0.3:
                img = self.rotate_image(img, angle)
                applied_deskew_angle = round(angle, 2)

        # 2. CLAHE (Contrast Enhancement)
        if self.enable_clahe:
            img = self.apply_clahe(img)
            clahe_applied = True

        # 3. Denoising (if configured)
        if self.enable_denoise:
            img = self.apply_denoising(img)
            denoise_applied = True

        return PreprocessResult(
            processed_image=img,
            deskew_angle=applied_deskew_angle,
            clahe_applied=clahe_applied,
            denoise_applied=denoise_applied,
        )
