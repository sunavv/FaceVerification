#!/usr/bin/env python3
"""
Explicit 1:1 Face Verification CLI Tool.
Compares a reference identity document face against a live/selfie face.

Usage:
    python verify.py --document <path_to_doc> --live <path_to_live> [--threshold 0.50] [--expected-name "NAME"]
"""

import sys
import argparse
from pathlib import Path

from app.utils.image import load_image_from_file
from app.services.verification_service import verification_service
from app.core.config import settings
from app.core.exceptions import AppException


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify a reference document face against a live face image (1:1 Age-Invariant Verification)."
    )
    parser.add_argument(
        "--document",
        type=str,
        required=True,
        help="Path to identity document image (JPG, PNG, WEBP)",
    )
    parser.add_argument(
        "--live",
        type=str,
        required=True,
        help="Path to live/selfie face image (JPG, PNG, WEBP)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.FACE_SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold (default: {settings.FACE_SIMILARITY_THRESHOLD})",
    )
    parser.add_argument(
        "--expected-name",
        type=str,
        default=None,
        help="Optional expected full name to verify against document OCR",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    doc_path = Path(args.document)
    live_path = Path(args.live)

    if not doc_path.exists():
        print(f"Error: Document file '{doc_path}' not found.", file=sys.stderr)
        sys.exit(1)

    if not live_path.exists():
        print(f"Error: Live image file '{live_path}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        doc_img = load_image_from_file(doc_path)
        live_img = load_image_from_file(live_path)

        result = verification_service.verify_direct_images(
            document_image=doc_img,
            live_image=live_img,
            expected_name=args.expected_name,
            threshold=args.threshold,
        )

        ref_detected = "DETECTED" if result["reference_face_detected"] else "NOT DETECTED"
        live_detected = "DETECTED" if result["live_face_detected"] else "NOT DETECTED"
        ref_quality = result.get("reference_face_quality", "GOOD")
        live_quality = result.get("live_face_quality", "GOOD")
        similarity = result["similarity"]
        threshold = result["threshold"]
        face_match_str = "TRUE" if result["face_match"] else "FALSE"

        print(f"REFERENCE FACE: {ref_detected}")
        print(f"LIVE FACE:      {live_detected}\n")
        print(f"REFERENCE QUALITY: {ref_quality}")
        print(f"LIVE QUALITY:      {live_quality}\n")
        if result.get("extracted_document_name"):
            print(f"DOCUMENT NAME:     {result['extracted_document_name']}")
        if args.expected_name:
            print(f"NAME MATCH:        {'TRUE' if result['name_match'] else 'FALSE'}")
        print(f"SIMILARITY: {similarity:.4f}")
        print(f"THRESHOLD:  {threshold:.4f}\n")
        print(f"FACE MATCH: {face_match_str}")
        print(f"FINAL VERIFIED: {'TRUE' if result['verified'] else 'FALSE'}")

        if result["verified"]:
            sys.exit(0)
        else:
            sys.exit(1)

    except AppException as e:
        print(f"\n[ERROR: {e.error_code.value}] {e.message}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"\n[ERROR] Verification error: {str(e)}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
