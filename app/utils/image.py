import io
import math
import base64
import cv2
import numpy as np
from PIL import Image, ImageOps
from typing import Tuple, Optional, Union, Dict, Any, List
from pathlib import Path
from app.core.exceptions import DocumentInvalidException
from app.core.logging import logger

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Standard ArcFace canonical 5-landmark reference coordinates on a 112x112 template
ARCFACE_CANONICAL_5KPS = np.array([
    [38.2946, 51.6963],  # Left Eye
    [73.5318, 51.5014],  # Right Eye
    [56.0252, 71.7366],  # Nose Tip
    [41.5493, 92.3655],  # Left Mouth Corner
    [70.7299, 92.2041],  # Right Mouth Corner
], dtype=np.float32)


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


def align_face_5landmarks(
    image: np.ndarray,
    landmarks: Union[list, np.ndarray],
    output_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """
    Perform 2D affine similarity transformation to align face landmarks to the
    canonical ArcFace 112x112 coordinate space.
    """
    if landmarks is None or len(landmarks) < 5:
        # Fallback to center crop if landmarks not available
        h, w = image.shape[:2]
        return cv2.resize(image, output_size)

    src_pts = np.array(landmarks[:5], dtype=np.float32)
    dst_pts = ARCFACE_CANONICAL_5KPS.copy()

    if output_size != (112, 112):
        scale_x = output_size[0] / 112.0
        scale_y = output_size[1] / 112.0
        dst_pts[:, 0] *= scale_x
        dst_pts[:, 1] *= scale_y

    # Estimate similarity transform matrix (rotation, scale, translation)
    transform_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if transform_matrix is None:
        # Fallback if matrix estimation fails
        return cv2.resize(image, output_size)

    aligned = cv2.warpAffine(
        image,
        transform_matrix,
        output_size,
        borderValue=0.0,
        flags=cv2.INTER_LINEAR,
    )
    return aligned


def normalize_face_lighting(aligned_face: np.ndarray) -> np.ndarray:
    """
    Apply subtle illumination normalization (CLAHE on L-channel) to minimize
    extreme contrast/lighting disparities between old documents and webcams,
    while preserving facial bone structure and biometric details.
    """
    if aligned_face is None or aligned_face.size == 0:
        return aligned_face

    try:
        lab = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply gentle CLAHE to luminance
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Blend original and equalized to prevent artifacting
        l_blended = cv2.addWeighted(l, 0.5, cl, 0.5, 0)
        
        merged_lab = cv2.merge((l_blended, a, b))
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return aligned_face


def enhance_image_for_detection(image: np.ndarray) -> np.ndarray:
    """
    Enhance low-contrast, noisy, or blurry scanned document images to maximize
    face detection recall on challenging identity documents.
    """
    if image is None or image.size == 0:
        return image
    try:
        # Convert to LAB to equalize lighting and contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        
        # Unsharp masking to accentuate subtle facial landmarks and edges on noisy/soft scans
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        unsharp = cv2.addWeighted(enhanced, 1.35, gaussian, -0.35, 0)
        return unsharp
    except Exception:
        return image


def assess_face_quality(
    image: np.ndarray,
    bbox: Union[list, np.ndarray],
    landmarks: Optional[Union[list, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Comprehensive face quality and usability assessment calibrated for both
    scanned/photographed ID documents and live webcam feeds:
    1. Size / Resolution: Minimum face size check (w, h >= 30px).
    2. Sharpness / Blur: Noise-tolerant Gaussian-smoothed Laplacian variance.
    3. Exposure / Luminance: Mean intensity and dynamic range.
    4. Pose: Yaw/pitch estimation from 5-point landmarks geometry.
    5. Frame Boundary Clipping: Checks if face is cut off by image borders.

    Returns:
        {
            "quality": "GOOD" | "ACCEPTABLE" | "POOR",
            "is_usable": bool,
            "blur_score": float,
            "luminance": float,
            "contrast_std": float,
            "face_width": int,
            "face_height": int,
            "estimated_yaw_ratio": float,
            "issues": List[str]
        }
    """
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    
    fw = max(0, x2 - x1)
    fh = max(0, y2 - y1)
    
    issues: List[str] = []
    
    # 1. Size check (allow small crops on full-page document scans down to 30px)
    if fw < 30 or fh < 30:
        issues.append("FACE_TOO_SMALL")

    # Crop raw face area
    face_crop = crop_face_bbox(image, bbox, padding_ratio=0.0)
    if face_crop.size == 0:
        return {
            "quality": "POOR",
            "is_usable": False,
            "blur_score": 0.0,
            "luminance": 0.0,
            "contrast_std": 0.0,
            "face_width": fw,
            "face_height": fh,
            "estimated_yaw_ratio": 1.0,
            "issues": ["EMPTY_FACE_CROP"],
        }

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    # 2. Blur / Sharpness check using noise-tolerant Laplacian variance
    # Mild smoothing prevents halftone scanner rosette noise from inflating blur metric
    denoised_gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
    laplacian_var = float(cv2.Laplacian(denoised_gray, cv2.CV_64F).var())
    
    if laplacian_var < 8.0:
        issues.append("FACE_TOO_BLURRY")
    elif laplacian_var < 35.0:
        issues.append("MILD_SCAN_SOFTNESS")

    # 3. Luminance & Exposure check
    mean_lum = float(np.mean(gray))
    std_contrast = float(np.std(gray))
    
    if mean_lum < 15.0:
        issues.append("FACE_TOO_DARK")
    elif mean_lum > 245.0:
        issues.append("FACE_OVEREXPOSED")
        
    if std_contrast < 10.0:
        issues.append("LOW_CONTRAST")

    # 4. Boundary Clipping check (is face cut off by sensor border?)
    if x1 <= 0 or y1 <= 0 or x2 >= img_w or y2 >= img_h:
        issues.append("FACE_CLIPPED_BY_FRAME")

    # 5. Pose estimation from 5 landmarks (if available)
    yaw_ratio = 1.0
    if landmarks is not None and len(landmarks) >= 5:
        # Landmark indices: 0: Left Eye, 1: Right Eye, 2: Nose, 3: Left Mouth, 4: Right Mouth
        pts = np.array(landmarks, dtype=np.float32)
        le_x = pts[0][0]
        re_x = pts[1][0]
        nose_x = pts[2][0]
        
        dist_left = abs(nose_x - le_x)
        dist_right = abs(re_x - nose_x)
        
        min_dist = max(min(dist_left, dist_right), 1e-4)
        max_dist = max(max(dist_left, dist_right), 1e-4)
        yaw_ratio = max_dist / min_dist

        if yaw_ratio > 3.8:
            issues.append("EXTREME_POSE_YAW")

    # Quality classification
    fatal_issues = {"FACE_TOO_SMALL", "FACE_TOO_BLURRY", "FACE_TOO_DARK", "FACE_OVEREXPOSED", "EXTREME_POSE_YAW", "EMPTY_FACE_CROP"}
    has_fatal = any(issue in fatal_issues for issue in issues)

    if not issues:
        quality_rating = "GOOD"
        is_usable = True
    elif not has_fatal:
        quality_rating = "ACCEPTABLE"
        is_usable = True
    else:
        quality_rating = "POOR"
        is_usable = False

    return {
        "quality": quality_rating,
        "is_usable": is_usable,
        "blur_score": round(laplacian_var, 2),
        "luminance": round(mean_lum, 2),
        "contrast_std": round(std_contrast, 2),
        "face_width": fw,
        "face_height": fh,
        "estimated_yaw_ratio": round(float(yaw_ratio), 2),
        "issues": issues,
    }


def encode_image_to_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode OpenCV BGR image to JPEG bytes."""
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise DocumentInvalidException("Failed to encode image to JPEG.")
    return buffer.tobytes()


def encode_image_to_base64_data_uri(image: np.ndarray, quality: int = 90) -> str:
    """Encode OpenCV BGR image to a base64 Data URI string (data:image/jpeg;base64,...)."""
    if image is None or image.size == 0:
        return ""
    jpeg_bytes = encode_image_to_jpeg(image, quality=quality)
    b64_str = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64_str}"


def enhance_and_upscale_face_crop(
    face_crop: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
    sharpen_factor: float = 0.4,
) -> np.ndarray:
    """
    Resolution adjustment & super-sampling for extracted document and live face crops:
    1. Upscales using high-order Lanczos-4 interpolation (cv2.INTER_LANCZOS4) to prevent pixelation.
    2. Applies bilateral filtering to suppress scanner halftone rosette noise while preserving facial contours.
    3. Uses unsharp masking for crisp landmark definition (eyes, nose, mouth corners).
    4. Applies gentle CLAHE contrast equalization in LAB luminance channel.
    """
    if face_crop is None or face_crop.size == 0:
        return face_crop

    try:
        # Step 1: Upscale to target high resolution
        upscaled = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_LANCZOS4)

        # Step 2: Edge-preserving noise suppression (halftone/compression artifact smoothing)
        bilateral = cv2.bilateralFilter(upscaled, d=5, sigmaColor=30, sigmaSpace=30)

        # Step 3: Unsharp masking for crisp landmark contours
        gaussian = cv2.GaussianBlur(bilateral, (0, 0), 1.5)
        sharpened = cv2.addWeighted(bilateral, 1.0 + sharpen_factor, gaussian, -sharpen_factor, 0)

        # Step 4: Contrast normalization in LAB space
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        l_blended = cv2.addWeighted(l, 0.65, cl, 0.35, 0)
        enhanced = cv2.cvtColor(cv2.merge((l_blended, a, b)), cv2.COLOR_LAB2BGR)
        return enhanced
    except Exception as e:
        logger.warning(f"Face crop upscaling/enhancement fallback: {str(e)}")
        if face_crop.shape[:2] != target_size:
            return cv2.resize(face_crop, target_size)
        return face_crop
