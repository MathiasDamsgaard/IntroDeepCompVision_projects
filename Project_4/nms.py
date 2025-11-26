import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from logger import logger
from matplotlib import patches
from model import get_model
from PIL import Image
from torch import nn
from torchmetrics.detection import MeanAveragePrecision
from torchvision import transforms
from torchvision.ops import nms
from tqdm import tqdm


def main() -> None:
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Path
    script_dir = Path(__file__).parent
    image_dir = "/dtu/datasets1/02516/potholes/images/"
    test_pickle = script_dir / "proposals_data/test_proposals.pkl"
    output_dir = script_dir / Path(args.output_dir)

    # Model
    model = get_model(num_classes=2)
    model = model.to(device)

    # Load best model
    model.load_state_dict(torch.load(output_dir / "best_model.pth"))
    model.eval()

    # Create output folder for NMS visualizations
    nms_output_dir = script_dir / "nms_results"
    nms_output_dir.mkdir(exist_ok=True)

    # Load test data
    with Path(test_pickle).open("rb") as f:
        data = pickle.load(f)  # noqa: S301

    # Setup transforms and softmax
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    softmax = nn.Softmax(dim=1)

    # Get unique image names
    unique_images = {item["image"] for item in data}

    # Storage for mAP computation
    all_predictions = []
    all_targets = []

    # For each image, get all proposals that belong to it
    with torch.no_grad():
        for image_name in tqdm(unique_images, desc="Processing images"):
            # Reverse lookup: get all objects matching this image key
            image_proposals = [item for item in data if item["image"] == image_name]

            # Load the image once
            img_path = Path(image_dir) / image_name
            img = Image.open(img_path).convert("RGB")

            # Storage for NMS per class
            # Class 0: background, Class 1: pothole
            boxes_by_class = {0: [], 1: []}
            scores_by_class = {0: [], 1: []}

            # Storage for ground truth boxes (potholes only - class 1)
            gt_boxes = []

            # Process each proposal
            for proposal in image_proposals:
                x, y, w, h = proposal["proposal"]

                # Crop and transform the proposal
                img_crop = img.crop((x, y, x + w, y + h))
                img_tensor = transform(img_crop).unsqueeze(0).to(device)  # type: ignore # noqa: PGH003

                # Get model prediction
                output = model(img_tensor)
                softmax_output = softmax(output)
                score, pred_class = torch.max(softmax_output, 1)

                # Convert box from (x, y, w, h) to (x1, y1, x2, y2) format for NMS
                box = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)
                pred_class_idx = pred_class.item()

                # Store box and score for the predicted class
                boxes_by_class[pred_class_idx].append(box)  # type: ignore  # noqa: PGH003
                scores_by_class[pred_class_idx].append(score.item())  # type: ignore  # noqa: PGH003

                # Store ground truth if it's a pothole
                if proposal["label"] == 1:
                    gt_boxes.append(box)

            # Run NMS for each class and store kept boxes/scores
            kept_boxes_by_class = {0: [], 1: []}
            kept_scores_by_class = {0: [], 1: []}

            for class_idx in [0, 1]:
                if len(boxes_by_class[class_idx]) > 0:
                    boxes = torch.stack(boxes_by_class[class_idx])
                    scores = torch.tensor(scores_by_class[class_idx])
                    keep_indices = nms(boxes, scores, iou_threshold=0.5)

                    # Filter boxes and scores with keep_indices
                    kept_boxes_by_class[class_idx] = boxes[keep_indices]  # type: ignore  # noqa: PGH003
                    kept_scores_by_class[class_idx] = scores[keep_indices]  # type: ignore  # noqa: PGH003

            # Plot and save the image with boxes
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(img)

            # Plot pothole detections (class 1) in red
            if len(kept_boxes_by_class[1]) > 0:
                for box, score in zip(kept_boxes_by_class[1], kept_scores_by_class[1], strict=True):
                    x1, y1, x2, y2 = box.tolist()
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none")
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 5, f"Pothole: {score:.2f}", color="red", fontsize=8)

            ax.set_title(f"NMS Results: {image_name}")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(nms_output_dir / f"{Path(image_name).stem}_nms.png", dpi=150)
            plt.close(fig)

            # Prepare predictions and targets for mAP (only for pothole class)
            if len(kept_boxes_by_class[1]) > 0:
                pred_dict = {
                    "boxes": kept_boxes_by_class[1],
                    "scores": kept_scores_by_class[1],
                    "labels": torch.ones(len(kept_boxes_by_class[1]), dtype=torch.int64),
                }
            else:
                pred_dict = {
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros(0),
                    "labels": torch.zeros(0, dtype=torch.int64),
                }
            all_predictions.append(pred_dict)

            if len(gt_boxes) > 0:
                target_dict = {
                    "boxes": torch.stack(gt_boxes),
                    "labels": torch.ones(len(gt_boxes), dtype=torch.int64),
                }
            else:
                target_dict = {
                    "boxes": torch.zeros((0, 4)),
                    "labels": torch.zeros(0, dtype=torch.int64),
                }
            all_targets.append(target_dict)

    # Compute mean average precision
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(all_predictions, all_targets)
    results = metric.compute()

    logger.info("Mean Average Precision Results:")
    logger.info(f"  mAP: {results['map']:.4f}")
    logger.info(f"  mAP@50: {results['map_50']:.4f}")
    logger.info(f"  mAP@75: {results['map_75']:.4f}")


if __name__ == "__main__":
    main()
