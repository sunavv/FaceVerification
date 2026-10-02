"""Synthetic test dataset generator for OCR module benchmarking and validation."""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_test_dataset(output_dir: str) -> None:
    """Generate benchmark document images for testing OCR performance across varying conditions.

    Generates:
    - clear_document.png: High resolution, crisp contrast
    - rotated_document.png: Rotated ~6 degrees to test deskewing
    - low_resolution_document.jpg: Low resolution (360x240)
    - low_lighting_document.jpg: Simulated uneven lighting/shadow gradient
    - ocr_fail_blank.png: Blank image with noise but no text
    - invalid_file.txt: Non-image file for validation testing
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Base Clear Document
    w, h = 1000, 650
    base_img = Image.new("RGB", (w, h), color=(250, 250, 252))
    draw = ImageDraw.Draw(base_img)

    # Draw header bar
    draw.rectangle([0, 0, w, 70], fill=(41, 128, 185))
    draw.rectangle([40, 90, w - 40, h - 40], outline=(180, 180, 190), width=2)

    # Draw Text using standard default font
    # Lines of realistic document
    lines = [
        ("OFFICIAL IDENTIFICATION DOCUMENT", (60, 25)),
        ("Name: Sunav Sharma", (70, 130)),
        ("Document No: ABC123456", (70, 180)),
        ("Date of Birth: 15/08/1990", (70, 230)),
        ("Nationality: Nepalese", (70, 280)),
        ("Issuing Authority: Department of Civil Records", (70, 330)),
        ("Expiry Date: 20/12/2030", (70, 380)),
        ("Status: ACTIVE VERIFIED", (70, 440)),
    ]

    for text, pos in lines:
        fill_color = (255, 255, 255) if pos[1] < 70 else (20, 25, 35)
        # Using PIL font if available or default
        draw.text(pos, text, fill=fill_color)

    # Save Clear Document
    clear_path = os.path.join(output_dir, "clear_document.png")
    base_img.save(clear_path, format="PNG")

    # Convert to OpenCV numpy array for OpenCV transforms
    cv_base = np.array(base_img)
    cv_base = cv2.cvtColor(cv_base, cv2.COLOR_RGB2BGR)

    # 2. Rotated Document (~5.5 degrees)
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, 5.5, 1.0)
    rotated_cv = cv2.warpAffine(
        cv_base, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(240, 240, 240)
    )
    cv2.imwrite(os.path.join(output_dir, "rotated_document.png"), rotated_cv)

    # 3. Low Resolution Document
    low_res = cv2.resize(cv_base, (360, 234), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(output_dir, "low_resolution_document.jpg"), low_res, [cv2.IMWRITE_JPEG_QUALITY, 40])

    # 4. Low Lighting / Shadow Gradient Document
    # Create diagonal gradient mask
    gradient = np.tile(np.linspace(0.3, 1.0, w, dtype=np.float32), (h, 1))
    gradient = np.stack([gradient, gradient, gradient], axis=2)
    # Add shadow on one corner
    dark_cv = (cv_base.astype(np.float32) * gradient).astype(np.uint8)
    # Add slight Gaussian noise
    noise = np.random.normal(0, 8, dark_cv.shape).astype(np.float32)
    dark_cv = np.clip(dark_cv.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "low_lighting_document.jpg"), dark_cv, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # 5. Blank / OCR Fail Document (plain noise background, no text)
    blank_cv = np.ones((400, 600, 3), dtype=np.uint8) * 245
    noise_blank = np.random.normal(0, 3, blank_cv.shape).astype(np.float32)
    blank_cv = np.clip(blank_cv.astype(np.float32) + noise_blank, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "ocr_fail_blank.png"), blank_cv)

    # 6. Invalid non-image file
    invalid_path = os.path.join(output_dir, "invalid_file.txt")
    with open(invalid_path, "w", encoding="utf-8") as f:
        f.write("This is a plain text file, not a valid document image.")


if __name__ == "__main__":
    generate_test_dataset(os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data"))
