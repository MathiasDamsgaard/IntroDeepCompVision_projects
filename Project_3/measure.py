# Evaluate segmentation performance metrics on predicted masks
# Calculates: Dice, IoU, Accuracy, Sensitivity, Specificity

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> dict:
    """Calculate segmentation metrics for binary masks.

    Args:
        pred: Predicted binary mask (values: 0 or 1)
        gt: Ground truth binary mask (values: 0 or 1)
        eps: Small epsilon to avoid division by zero

    Returns:
        Dictionary with metrics: dice, iou, accuracy, sensitivity, specificity

    """
    # Flatten arrays
    pred = pred.flatten()
    gt = gt.flatten()

    # Calculate confusion matrix components
    tp = np.sum((pred == 1) & (gt == 1))  # True Positives
    tn = np.sum((pred == 0) & (gt == 0))  # True Negatives
    fp = np.sum((pred == 1) & (gt == 0))  # False Positives
    fn = np.sum((pred == 0) & (gt == 1))  # False Negatives

    # Dice Coefficient (F1-score)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    # Intersection over Union (IoU / Jaccard Index)
    iou = tp / (tp + fp + fn + eps)

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    # Sensitivity (Recall / True Positive Rate)
    sensitivity = tp / (tp + fn + eps)

    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp + eps)

    return {
        "dice": dice,
        "iou": iou,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions")
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Directory containing predictions (with predictions.npy and ground_truths.npy)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output CSV file for results (default: <pred_dir>/metrics.csv)"
    )
    args = parser.parse_args()

    # Load predictions and ground truths
    pred_dir = Path(args.pred_dir)
    pred_path = pred_dir / "predictions.npy"
    gt_path = pred_dir / "ground_truths.npy"

    if not pred_path.exists():
        msg = f"Predictions not found: {pred_path}"
        raise FileNotFoundError(msg)
    if not gt_path.exists():
        msg = f"Ground truths not found: {gt_path}"
        raise FileNotFoundError(msg)

    predictions = np.load(pred_path)
    ground_truths = np.load(gt_path)

    # Calculate metrics for each image
    all_metrics = []
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths, strict=False)):
        metrics = calculate_metrics(pred, gt)
        metrics["image_id"] = i
        all_metrics.append(metrics)

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)

    # Calculate mean and std
    mean_metrics = df[["dice", "iou", "accuracy", "sensitivity", "specificity"]].mean()
    std_metrics = df[["dice", "iou", "accuracy", "sensitivity", "specificity"]].std()

    # Print results

    # Save detailed results
    output_file = Path(args.output) if args.output else pred_dir / "metrics.csv"
    df.to_csv(output_file, index=False)

    # Save summary
    summary_file = output_file.with_name(output_file.stem + "_summary.csv")
    summary_df = pd.DataFrame(
        {
            "metric": ["dice", "iou", "accuracy", "sensitivity", "specificity"],
            "mean": [
                mean_metrics["dice"],
                mean_metrics["iou"],
                mean_metrics["accuracy"],
                mean_metrics["sensitivity"],
                mean_metrics["specificity"],
            ],
            "std": [
                std_metrics["dice"],
                std_metrics["iou"],
                std_metrics["accuracy"],
                std_metrics["sensitivity"],
                std_metrics["specificity"],
            ],
        }
    )
    summary_df.to_csv(summary_file, index=False)

    # Save summary as JSON
    summary_json = {
        "num_images": len(predictions),
        "mean_dice": float(mean_metrics["dice"]),
        "std_dice": float(std_metrics["dice"]),
        "mean_iou": float(mean_metrics["iou"]),
        "std_iou": float(std_metrics["iou"]),
        "mean_accuracy": float(mean_metrics["accuracy"]),
        "std_accuracy": float(std_metrics["accuracy"]),
        "mean_sensitivity": float(mean_metrics["sensitivity"]),
        "std_sensitivity": float(std_metrics["sensitivity"]),
        "mean_specificity": float(mean_metrics["specificity"]),
        "std_specificity": float(std_metrics["specificity"]),
    }
    json_file = output_file.with_name(output_file.stem + "_summary.json")
    json_file.write_text(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
