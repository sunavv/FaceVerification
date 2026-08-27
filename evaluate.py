#!/usr/bin/env python3
"""
Benchmark Evaluation Script for Face Verification.
Evaluates positive (genuine) and negative (impostor) image pairs.

Expected dataset directory structure:
dataset_dir/
├── genuine/
│   ├── pair_001_doc.jpg
│   ├── pair_001_live.jpg
│   ├── ...
└── impostor/
    ├── pair_001_doc.jpg
    ├── pair_001_live.jpg
    └── ...

Or a pairs.csv / pairs.json file specifying doc_path, live_path, is_genuine.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

from app.utils.image import load_image_from_file
from app.services.face_service import face_service
from app.core.config import settings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate face verification accuracy, FAR, and FRR across genuine and impostor pairs."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./data/benchmark",
        help="Path to evaluation dataset containing genuine/ and impostor/ subdirectories.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.FACE_SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold (default: {settings.FACE_SIMILARITY_THRESHOLD})",
    )
    return parser.parse_args()


def load_pairs_from_dir(dataset_dir: Path) -> List[Tuple[Path, Path, bool]]:
    """Discover genuine and impostor image pairs from directory structure."""
    pairs = []

    # 1. Genuine pairs
    genuine_dir = dataset_dir / "genuine"
    if genuine_dir.exists():
        docs = sorted(list(genuine_dir.glob("*_doc.*")) + list(genuine_dir.glob("*_a.*")))
        for doc in docs:
            prefix = doc.stem.replace("_doc", "").replace("_a", "")
            matches = list(genuine_dir.glob(f"{prefix}_live.*")) + list(genuine_dir.glob(f"{prefix}_b.*"))
            if matches:
                pairs.append((doc, matches[0], True))

    # 2. Impostor pairs
    impostor_dir = dataset_dir / "impostor"
    if impostor_dir.exists():
        docs = sorted(list(impostor_dir.glob("*_doc.*")) + list(impostor_dir.glob("*_a.*")))
        for doc in docs:
            prefix = doc.stem.replace("_doc", "").replace("_a", "")
            matches = list(impostor_dir.glob(f"{prefix}_live.*")) + list(impostor_dir.glob(f"{prefix}_b.*"))
            if matches:
                pairs.append((doc, matches[0], False))

    return pairs


def run_evaluation(pairs: List[Tuple[Path, Path, bool]], threshold: float) -> Dict[str, Any]:
    genuine_sims: List[float] = []
    impostor_sims: List[float] = []

    tp = 0  # Genuine classified as Genuine (sim >= thresh)
    fn = 0  # Genuine classified as Impostor (sim < thresh) -> False Rejection
    tn = 0  # Impostor classified as Impostor (sim < thresh)
    fp = 0  # Impostor classified as Genuine (sim >= thresh) -> False Acceptance

    print(f"\nProcessing {len(pairs)} pairs with threshold={threshold:.4f}...")

    for doc_p, live_p, is_genuine in pairs:
        try:
            doc_img = load_image_from_file(doc_p)
            live_img = load_image_from_file(live_p)

            doc_face = face_service.process_document_face(doc_img)
            live_face = face_service.process_live_face(live_img)

            sim_data = face_service.compute_cosine_similarity(
                doc_face["embedding"],
                live_face["embedding"],
                threshold=threshold,
            )
            sim = sim_data["similarity"]
            predicted_genuine = sim >= threshold

            if is_genuine:
                genuine_sims.append(sim)
                if predicted_genuine:
                    tp += 1
                else:
                    fn += 1
            else:
                impostor_sims.append(sim)
                if predicted_genuine:
                    fp += 1
                else:
                    tn += 1
        except Exception as e:
            print(f"Skipping pair ({doc_p.name}, {live_p.name}) due to error: {e}", file=sys.stderr)

    total_genuine = len(genuine_sims)
    total_impostor = len(impostor_sims)
    total_evaluated = total_genuine + total_impostor

    avg_genuine_sim = float(np.mean(genuine_sims)) if genuine_sims else 0.0
    avg_impostor_sim = float(np.mean(impostor_sims)) if impostor_sims else 0.0

    accuracy = (tp + tn) / max(total_evaluated, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    # FAR: False Acceptance Rate = FP / (FP + TN)
    far = fp / max(total_impostor, 1)
    # FRR: False Rejection Rate = FN / (FN + TP)
    frr = fn / max(total_genuine, 1)

    return {
        "num_genuine_pairs": total_genuine,
        "num_impostor_pairs": total_impostor,
        "avg_genuine_similarity": round(avg_genuine_sim, 4),
        "avg_impostor_similarity": round(avg_impostor_sim, 4),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_acceptance_rate": round(far, 4),
        "false_rejection_rate": round(frr, 4),
    }


def print_report(metrics: Dict[str, Any], threshold: float):
    print("=" * 60)
    print("      FACE RECOGNITION BENCHMARK EVALUATION REPORT")
    print("=" * 60)
    print(f"Evaluated Threshold:             {threshold:.4f}")
    print(f"Genuine Pairs Count:             {metrics['num_genuine_pairs']}")
    print(f"Impostor Pairs Count:            {metrics['num_impostor_pairs']}")
    print("-" * 60)
    print(f"Average Genuine Similarity:      {metrics['avg_genuine_similarity']:.4f}")
    print(f"Average Impostor Similarity:     {metrics['avg_impostor_similarity']:.4f}")
    print("-" * 60)
    print(f"True Positives  (TP):            {metrics['true_positives']}")
    print(f"True Negatives  (TN):            {metrics['true_negatives']}")
    print(f"False Positives (FP):            {metrics['false_positives']}")
    print(f"False Negatives (FN):            {metrics['false_negatives']}")
    print("-" * 60)
    print(f"Accuracy:                        {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:                       {metrics['precision'] * 100:.2f}%")
    print(f"Recall / TPR:                    {metrics['recall'] * 100:.2f}%")
    print(f"False Acceptance Rate (FAR):     {metrics['false_acceptance_rate'] * 100:.2f}%")
    print(f"False Rejection Rate  (FRR):     {metrics['false_rejection_rate'] * 100:.2f}%")
    print("=" * 60)
    print("\nNote: Calibrate the similarity threshold on representative target datasets to balance FAR vs FRR.")


def main():
    args = parse_args()
    ds_dir = Path(args.dataset_dir)

    if not ds_dir.exists():
        print(f"Dataset directory '{ds_dir}' not found. Please provide a valid directory.", file=sys.stderr)
        sys.exit(1)

    pairs = load_pairs_from_dir(ds_dir)
    if not pairs:
        print(f"No image pairs found in '{ds_dir}'. Make sure 'genuine/' and 'impostor/' folders contain pairs.", file=sys.stderr)
        sys.exit(1)

    metrics = run_evaluation(pairs, args.threshold)
    print_report(metrics, args.threshold)


if __name__ == "__main__":
    main()
