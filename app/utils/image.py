import io
import cv2
import numpy as np
from PIL import Image, ImageOps
from typing import Tuple, Optional, Union
from pathlib import Path
from app.core.exceptions import DocumentInvalidException
from app.core.logging import logger

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def validate_image_extension(filename: str) -> None:
    """Validate file extension for allowed formats."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise DocumentInvalidException(
            f"Unsupported file extension '{ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into an OpenCV BGR numpy array with EXIF orientation correction."""
    if not image_bytes or len(image_bytes) == 0:
        raise DocumentInvalidException("Uploaded file contains empty or zero bytes.")

    try:
        # Use PIL to read and automatically correct EXIF rotation
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image = ImageOps.exif_transpose(pil_image)
        
        # Convert to RGB (in case of RGBA, Palette, or Grayscale)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
            
        rgb_arr = np.array(pil_image)
        # Convert RGB to OpenCV BGR
        bgr_image = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        
        if bgr_image is None or bgr_image.size == 0:
            raise DocumentInvalidException("Failed to decode image pixels.")
            
        return bgr_image
    except DocumentInvalidException:
        raise
    except Exception as e:
        logger.error(f"Image decode error: {str(e)}")
        raise DocumentInvalidException(f"Invalid or corrupted image data: {str(e)}")


def load_image_from_file(file_path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk into an OpenCV BGR numpy array."""
    p = Path(file_path)
    if not p.exists():
        raise DocumentInvalidException(f"Image file not found: {p}")
    try:
        with open(p, "rb") as f:
            return decode_image_bytes(f.read())
    except Exception as e:
        raise DocumentInvalidException(f"Failed to read image at {file_path}: {str(e)}")


def crop_face_bbox(image: np.ndarray, bbox: Union[list, np.ndarray], padding_ratio: float = 0.15) -> np.ndarray:
    """Crop the face bounding box with optional padding and edge clamping."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    
    bw = x2 - x1
    bh = y2 - y1
    
    pad_w = int(bw * padding_ratio)
    pad_h = int(bh * padding_ratio)
    
    cx1 = max(0, x1 - pad_w)
    cy1 = max(0, y1 - pad_h)
    cx2 = min(w, x2 + pad_w)
    cy2 = min(h, y2 + pad_h)
    
    return image[cy1:cy2, cx1:cx2].copy()


def encode_image_to_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode OpenCV BGR image to JPEG bytes."""
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise DocumentInvalidException("Failed to encode image to JPEG.")
    return buffer.tobytes()
