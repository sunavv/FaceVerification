#!/usr/bin/env python3
"""
Benchmark Evaluation Script for Age-Invariant Face Verification.
Evaluates genuine (same person across different ages) vs impostor (different people) pairs.

Supported dataset directory structures:

Structure A:
dataset_dir/
├── same_person/
│   ├── person_a_young.jpg
│   ├── person_a_current.jpg
│   ├── person_a_old.jpg
│   ├── person_b_young.jpg
│   └── person_b_current.jpg
└── different_people/
    ├── person_a.jpg
    ├── person_b.jpg
    ├── person_c.jpg
    └── ...

Structure B:
dataset_dir/
├── genuine/
│   ├── pair1_doc.jpg, pair1_live.jpg
│   └── ...
└── impostor/
    ├── pair1_doc.jpg, pair1_live.jpg
    └── ...
"""

import sys
import itertools
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

from app.utils.image import load_image_from_file
from app.services.face_recognition_service import face_recognition_service
from app.core.config import settings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate age-invariant face verification accuracy, FAR, and FRR across genuine and impostor pairs."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./tests/data",
        help="Path to evaluation dataset directory (containing same_person/ & different_people/ or genuine/ & impostor/).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.FACE_SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold (default: {settings.FACE_SIMILARITY_THRESHOLD})",
    )
    return parser.parse_args()


def load_pairs_from_directory(dataset_dir: Path) -> List[Tuple[Path, Path, bool, str]]:
    """
    Discover genuine pairs (same person at different ages) and impostor pairs (different people).
    Returns list of (img1_path, img2_path, is_genuine, pair_label).
    """
    pairs: List[Tuple[Path, Path, bool, str]] = []

    # Check for same_person & different_people structure
    same_person_dir = dataset_dir / "same_person"
    diff_people_dir = dataset_dir / "different_people"

    if same_person_dir.exists():
        # Group photos by person prefix (e.g. 'person_a_old.jpg' and 'person_a_young.jpg' -> group 'person_a')
        person_groups: Dict[str, List[Path]] = {}
        for img_file in same_person_dir.glob("*.*"):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                # Extract person identity name (everything before _young, _old, _current, _doc, _live, etc.)
                name_parts = img_file.stem.split("_")
                if len(name_parts) >= 2 and name_parts[-1] in {"young", "old", "current", "doc", "live", "a", "b", "1", "2"}:
                    person_id = "_".join(name_parts[:-1])
                else:
                    person_id = name_parts[0]
                person_groups.setdefault(person_id, []).append(img_file)

        # Generate all 2-combinations within each person (Genuine pairs across ages)
        for pid, files in person_groups.items():
            if len(files) >= 2:
                for f1, f2 in itertools.combinations(files, 2):
                    pairs.append((f1, f2, True, f"Same Person ({pid}) [{f1.name} vs {f2.name}]"))

    if diff_people_dir.exists():
        diff_files = [
            f for f in diff_people_dir.glob("*.*")
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        # Generate combinations between different people (Impostor pairs)
        for f1, f2 in itertools.combinations(diff_files, 2):
            pairs.append((f1, f2, False, f"Different People [{f1.name} vs {f2.name}]"))

    # Also support standard genuine/ and impostor/ subdirs
    gen_dir = dataset_dir / "genuine"
    imp_dir = dataset_dir / "impostor"

    if gen_dir.exists():
        docs = sorted(list(gen_dir.glob("*_doc.*")) + list(gen_dir.glob("*_a.*")))
        for doc in docs:
            prefix = doc.stem.replace("_doc", "").replace("_a", "")
            matches = list(gen_dir.glob(f"{prefix}_live.*")) + list(gen_dir.glob(f"{prefix}_b.*"))
            if matches:
                pairs.append((doc, matches[0], True, f"Genuine ({prefix})"))

    if imp_dir.exists():
        docs = sorted(list(imp_dir.glob("*_doc.*")) + list(imp_dir.glob("*_a.*")))
        for doc in docs:
            prefix = doc.stem.replace("_doc", "").replace("_a", "")
            matches = list(imp_dir.glob(f"{prefix}_live.*")) + list(imp_dir.glob(f"{prefix}_b.*"))
            if matches:
                pairs.append((doc, matches[0], False, f"Impostor ({prefix})"))

    return pairs


def run_benchmark(pairs: List[Tuple[Path, Path, bool, str]], threshold: float) -> Dict[str, Any]:
    genuine_sims: List[float] = []
    impostor_sims: List[float] = []

    tp = 0  # Genuine classified as Genuine (sim >= thresh)
    fn = 0  # Genuine classified as Impostor (sim < thresh) -> False Rejection
    tn = 0  # Impostor classified as Impostor (sim < thresh)
    fp = 0  # Impostor classified as Genuine (sim >= thresh) -> False Acceptance

    print(f"\nRunning benchmark on {len(pairs)} image pairs at threshold={threshold:.4f}...\n")
    print(f"{'TYPE':<12} | {'SIMILARITY':<10} | {'PREDICTED':<10} | {'GROUND TRUTH':<12} | PAIR")
    print("-" * 80)

    for img1_p, img2_p, is_genuine, label in pairs:
        try:
            img1 = load_image_from_file(img1_p)
            img2 = load_image_from_file(img2_p)

            face1 = face_recognition_service.process_document_face(img1, enforce_quality=False)
            face2 = face_recognition_service.process_live_face(img2, enforce_quality=False)

            sim_data = face_recognition_service.compute_cosine_similarity(
                face1["embedding"],
                face2["embedding"],
                threshold=threshold,
            )
            sim = sim_data["similarity"]
            predicted_match = sim >= threshold

            gt_str = "GENUINE" if is_genuine else "IMPOSTOR"
            pred_str = "MATCH" if predicted_match else "NO MATCH"
            pair_type = "GENUINE" if is_genuine else "IMPOSTOR"

            print(f"{pair_type:<12} | {sim:<10.4f} | {pred_str:<10} | {gt_str:<12} | {label}")

            if is_genuine:
                genuine_sims.append(sim)
                if predicted_match:
                    tp += 1
                else:
                    fn += 1
            else:
                impostor_sims.append(sim)
                if predicted_match:
                    fp += 1
                else:
                    tn += 1
        except Exception as e:
            print(f"Skipping pair ({img1_p.name}, {img2_p.name}): {e}", file=sys.stderr)

    total_genuine = len(genuine_sims)
    total_impostor = len(impostor_sims)
    total_evaluated = total_genuine + total_impostor

    avg_genuine_sim = float(np.mean(genuine_sims)) if genuine_sims else 0.0
    min_genuine_sim = float(np.min(genuine_sims)) if genuine_sims else 0.0
    max_genuine_sim = float(np.max(genuine_sims)) if genuine_sims else 0.0

    avg_impostor_sim = float(np.mean(impostor_sims)) if impostor_sims else 0.0
    max_impostor_sim = float(np.max(impostor_sims)) if impostor_sims else 0.0

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
        "genuine_sim_mean": round(avg_genuine_sim, 4),
        "genuine_sim_min": round(min_genuine_sim, 4),
        "genuine_sim_max": round(max_genuine_sim, 4),
        "impostor_sim_mean": round(avg_impostor_sim, 4),
        "impostor_sim_max": round(max_impostor_sim, 4),
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
    print("\n" + "=" * 65)
    print("      AGE-INVARIANT FACE VERIFICATION BENCHMARK REPORT")
    print("=" * 65)
    print(f"Evaluated Threshold:             {threshold:.4f}")
    print(f"Genuine Pairs Count:             {metrics['num_genuine_pairs']}")
    print(f"Impostor Pairs Count:            {metrics['num_impostor_pairs']}")
    print("-" * 65)
    print(f"Genuine Similarity Mean:         {metrics['genuine_sim_mean']:.4f}")
    print(f"Genuine Similarity Minimum:      {metrics['genuine_sim_min']:.4f}")
    print(f"Genuine Similarity Maximum:      {metrics['genuine_sim_max']:.4f}")
    print("-" * 65)
    print(f"Impostor Similarity Mean:        {metrics['impostor_sim_mean']:.4f}")
    print(f"Impostor Similarity Maximum:     {metrics['impostor_sim_max']:.4f}")
    print("-" * 65)
    print(f"True Positives  (TP):            {metrics['true_positives']}")
    print(f"True Negatives  (TN):            {metrics['true_negatives']}")
    print(f"False Positives (FP):            {metrics['false_positives']}")
    print(f"False Negatives (FN):            {metrics['false_negatives']}")
    print("-" * 65)
    print(f"Accuracy:                        {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:                       {metrics['precision'] * 100:.2f}%")
    print(f"Recall / TPR:                    {metrics['recall'] * 100:.2f}%")
    print(f"False Acceptance Rate (FAR):     {metrics['false_acceptance_rate'] * 100:.2f}%")
    print(f"False Rejection Rate  (FRR):     {metrics['false_rejection_rate'] * 100:.2f}%")
    print("=" * 65 + "\n")


def main():
    args = parse_args()
    ds_dir = Path(args.dataset_dir)

    if not ds_dir.exists():
        print(f"Error: Dataset directory '{ds_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    pairs = load_pairs_from_directory(ds_dir)
    if not pairs:
        print(f"Error: No image pairs found in '{ds_dir}'.", file=sys.stderr)
        sys.exit(1)

    metrics = run_benchmark(pairs, args.threshold)
    print_report(metrics, args.threshold)


if __name__ == "__main__":
    main()
