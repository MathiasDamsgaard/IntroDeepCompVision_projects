# Evaluate segmentation performance metrics on predicted masks
# Calculates: Dice, IoU, Accuracy, Sensitivity, Specificity

import argparse
from pathlib import Path

import numpy as np

NUM_PARTS = 3


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
    }


def parse_experiment_name(pred_dir: str) -> dict:
    """Parse experiment name to extract dataset, model, and loss."""
    name = Path(pred_dir).name
    parts = name.split("_")

    if len(parts) >= NUM_PARTS:
        return {
            "dataset": parts[0].capitalize(),
            "model": parts[1].capitalize(),
            "loss": "_".join(parts[2:]).capitalize(),
        }
    return {"dataset": "Unknown", "model": "Unknown", "loss": "Unknown"}


def measure_single(pred_dir: str) -> dict:
    """Measure metrics for a single prediction directory."""
    pred_dir = Path(pred_dir)
    pred_path = pred_dir / "predictions.npy"
    gt_path = pred_dir / "ground_truths.npy"

    if not pred_path.exists() or not gt_path.exists():
        return None

    predictions = np.load(pred_path)
    ground_truths = np.load(gt_path)

    # Calculate metrics for each image
    all_dice = []
    all_iou = []
    all_accuracy = []
    all_sensitivity = []
    all_specificity = []

    for pred, gt in zip(predictions, ground_truths, strict=False):
        metrics = calculate_metrics(pred, gt)
        all_dice.append(metrics["dice"])
        all_iou.append(metrics["iou"])
        all_accuracy.append(metrics["accuracy"])
        all_sensitivity.append(metrics["sensitivity"])
        all_specificity.append(metrics["specificity"])

    # Calculate mean and std
    info = parse_experiment_name(str(pred_dir))
    return {
        "dataset": info["dataset"],
        "model": info["model"],
        "loss": info["loss"],
        "n_images": len(predictions),
        "dice_mean": np.mean(all_dice),
        "dice_std": np.std(all_dice),
        "iou_mean": np.mean(all_iou),
        "iou_std": np.std(all_iou),
        "accuracy_mean": np.mean(all_accuracy),
        "accuracy_std": np.std(all_accuracy),
        "sensitivity_mean": np.mean(all_sensitivity),
        "sensitivity_std": np.std(all_sensitivity),
        "specificity_mean": np.mean(all_specificity),
        "specificity_std": np.std(all_specificity),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions")
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=None,
        help="Model name (e.g., 'drive_unet_crossentropyloss') - prepends 'Project_3/dataset/predictions/'",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Measure all prediction directories in Project_3/dataset/predictions/",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Project_3/dataset/metrics_summary.txt",
        help="Output text file for results",
    )
    args = parser.parse_args()

    results = []

    if args.all:
        # Find all prediction directories under dataset/predictions/
        pred_dirs = sorted(Path("Project_3/dataset/predictions").glob("*_*_*"))

        for pred_dir in pred_dirs:
            if pred_dir.is_dir():
                result = measure_single(str(pred_dir))
                if result:
                    results.append(result)
                else:
                    pass

    elif args.pred_dir:
        # Automatically prepend the dataset/predictions path
        pred_dir_path = Path("Project_3/dataset/predictions") / args.pred_dir
        result = measure_single(str(pred_dir_path))
        if result:
            results.append(result)
        else:
            return

    else:
        parser.print_help()
        return

    # Write results to text file
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with Path.open(output_file, "w") as f:
        f.write("=" * 90 + "\n")
        f.write("SEGMENTATION METRICS SUMMARY\n")
        f.write("=" * 90 + "\n\n")

        for result in results:
            f.write(f"Dataset: {result['dataset']:<10} Model: {result['model']:<10} Loss: {result['loss']}\n")
            f.write("-" * 90 + "\n")
            f.write(f"  Images:      {result['n_images']}\n")
            f.write(f"  Dice:        {result['dice_mean']:.4f} ± {result['dice_std']:.4f}\n")
            f.write(f"  IoU:         {result['iou_mean']:.4f} ± {result['iou_std']:.4f}\n")
            f.write(f"  Accuracy:    {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}\n")
            f.write(f"  Sensitivity: {result['sensitivity_mean']:.4f} ± {result['sensitivity_std']:.4f}\n")
            f.write(f"  Specificity: {result['specificity_mean']:.4f} ± {result['specificity_std']:.4f}\n")
            f.write("\n")

        f.write("=" * 90 + "\n")


if __name__ == "__main__":
    main()
