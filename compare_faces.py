#!/usr/bin/env python3
"""
CLI Utility to compare two static images without using the webcam.
Usage:
    python compare_faces.py <document_image> <live_image> [--threshold 0.40] [--expected-name "NAME"]
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
        description="Compare document image against a live/selfie image using InsightFace facial embeddings."
    )
    parser.add_argument("document", type=str, help="Path to document image file (JPG, PNG, WEBP)")
    parser.add_argument("live", type=str, help="Path to live/selfie image file (JPG, PNG, WEBP)")
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
        help="Optional expected name to verify against OCR-extracted name",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    doc_path = Path(args.document)
    live_path = Path(args.live)

    if not doc_path.exists():
        print(f"Error: Document file '{doc_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not live_path.exists():
        print(f"Error: Live image file '{live_path}' does not exist.", file=sys.stderr)
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

        doc_detected_str = "YES" if result["document_face_detected"] else "NO"
        live_detected_str = "YES" if result["live_face_detected"] else "NO"
        result_str = "MATCH" if result["face_match"] else "NO MATCH"

        print(f"Document face detected: {doc_detected_str}")
        print(f"Live face detected:     {live_detected_str}")
        print(f"Document faces:         {result['document_face_count']}")
        print(f"Live faces:             {result['live_face_count']}")
        if result.get("extracted_document_name"):
            print(f"Extracted document name: {result['extracted_document_name']}")
        if args.expected_name:
            print(f"Expected name:          {args.expected_name}")
            print(f"Name match:             {'YES' if result['name_match'] else 'NO'}")
        print(f"Cosine similarity:      {result['similarity']:.4f}")
        print(f"Threshold:              {result['threshold']:.4f}")
        print(f"Result:                 {result_str}")
        print(f"Final Verification:     {'VERIFIED' if result['verified'] else 'NOT VERIFIED'}")

        if result["verified"]:
            sys.exit(0)
        else:
            sys.exit(1)

    except AppException as e:
        print(f"\n[ERROR: {e.error_code.value}] {e.message}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {str(e)}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
