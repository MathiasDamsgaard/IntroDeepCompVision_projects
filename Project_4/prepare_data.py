import pickle
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import xmltodict
from logger import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def run_selective_search(image: np.ndarray, method: str = "fast") -> np.ndarray:
    """Run Selective Search on an image to extract object proposals.

    Args:
        image (numpy.ndarray): The input image.
        method (str): The method to use, either "fast" or "quality".

    Returns:
        numpy.ndarray: A list of bounding boxes in [x, y, w, h] format.

    """
    # Create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # Set the image on which we will run segmentation
    ss.setBaseImage(image)

    # Switch to fast but low recall Selective Search method
    if method == "fast":
        ss.switchToSelectiveSearchFast()
    # Switch to high recall but slow Selective Search method
    elif method == "quality":
        ss.switchToSelectiveSearchQuality()

    # Run selective search segmentation on input image
    return np.array(ss.process())


def compute_iou(box_a: list[int], box_b: list[int]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.

    Boxes are expected in [xmin, ymin, xmax, ymax] format.

    Args:
        box_a (list[int]): First bounding box [xmin, ymin, xmax, ymax].
        box_b (list[int]): Second bounding box [xmin, ymin, xmax, ymax].

    Returns:
        float: The IoU score.

    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    return inter_area / float(box_a_area + box_b_area - inter_area)


def evaluate_recall(
    image_list: list[str],
    proposals_dict: dict[str, Any],
    annotations_dict: dict[str, list[list[int]]],
    k_values: list[int],
    iou_threshold: float = 0.5,
) -> dict[int, float]:
    """Evaluate the recall of proposals at different k values.

    Args:
        image_list (list[str]): List of image names to evaluate.
        proposals_dict (dict[str, Any]): Dictionary mapping image name to list of proposals [x, y, w, h].
        annotations_dict (dict[str, list[list[int]]]): Dictionary mapping image name to list of ground
            truth boxes [xmin, ymin, xmax, ymax].
        k_values (list[int]): List of integers representing the number of top proposals to consider.
        iou_threshold (float): IoU threshold to consider a ground truth covered. Defaults to 0.5.

    Returns:
        dict[int, float]: Average recall for each k.

    """
    recalls = {k: [] for k in k_values}

    for img_name in image_list:
        gt_boxes = annotations_dict[img_name]
        if len(gt_boxes) == 0:
            continue

        # Proposals are [x, y, w, h]
        proposals = proposals_dict[img_name]

        # Convert to [xmin, ymin, xmax, ymax]
        proposals_xyxy = [[p[0], p[1], p[0] + p[2], p[1] + p[3]] for p in proposals]

        for k in k_values:
            current_proposals = proposals_xyxy[:k]
            found_gt = 0
            for gt in gt_boxes:
                # Check if this GT is covered by any proposal with IoU >= iou_threshold
                covered = False
                for prop in current_proposals:
                    if compute_iou(gt, prop) >= iou_threshold:
                        covered = True
                        break
                if covered:
                    found_gt += 1

            recall = found_gt / len(gt_boxes)
            recalls[k].append(recall)

    return {k: float(np.mean(v)) for k, v in recalls.items()}


def evaluate_mabo(
    image_list: list[str],
    proposals_dict: dict[str, Any],
    annotations_dict: dict[str, list[list[int]]],
    k_values: list[int],
) -> dict[int, float]:
    """Evaluate the Mean Average Best Overlap (MABO) of proposals at different k values.

    Args:
        image_list (list[str]): List of image names to evaluate.
        proposals_dict (dict[str, Any]): Dictionary mapping image name to list of proposals [x, y, w, h].
        annotations_dict (dict[str, list[list[int]]]): Dictionary mapping image name to list of ground
            truth boxes [xmin, ymin, xmax, ymax].
        k_values (list[int]): List of integers representing the number of top proposals to consider.

    Returns:
        dict[int, float]: Average MABO for each k.

    """
    mabos = {k: [] for k in k_values}

    for img_name in image_list:
        gt_boxes = annotations_dict[img_name]
        if len(gt_boxes) == 0:
            continue

        # Proposals are [x, y, w, h]
        proposals = proposals_dict[img_name]

        # Convert to [xmin, ymin, xmax, ymax]
        proposals_xyxy = [[p[0], p[1], p[0] + p[2], p[1] + p[3]] for p in proposals]

        for k in k_values:
            current_proposals = proposals_xyxy[:k]
            if not current_proposals:
                mabos[k].append(0.0)
                continue

            best_overlaps = []
            for gt in gt_boxes:
                max_iou = 0.0
                for prop in current_proposals:
                    iou = compute_iou(gt, prop)
                    max_iou = max(max_iou, iou)
                best_overlaps.append(max_iou)

            # Average Best Overlap for this image
            abo = np.mean(best_overlaps)
            mabos[k].append(abo)

    return {k: float(np.mean(v)) for k, v in mabos.items()}


def prepare_dataset(
    image_names: list[str],
    all_proposals: dict[str, Any],
    all_annotations: dict[str, list[list[int]]],
    selected_k: int,
    iou_threshold: float,
    max_negatives: int = 50,
) -> list[dict[str, Any]]:
    """Prepare the dataset by assigning labels to proposals based on IoU with ground truth.

    Args:
        image_names (list[str]): List of image names to include.
        all_proposals (dict[str, Any]): Dictionary of proposals.
        all_annotations (dict[str, list[list[int]]]): Dictionary of annotations.
        selected_k (int): Number of top proposals to select.
        iou_threshold (float): IoU threshold for positive label.
        max_negatives (int): Maximum number of negative samples to keep per image.

    Returns:
        list[dict[str, Any]]: List of samples with image name, proposal, label, and IoU.

    """
    dataset = []
    for img_name in image_names:
        gt_boxes = all_annotations[img_name]
        proposals = all_proposals[img_name][:selected_k]

        neg_count = 0
        for p in proposals:
            # p is [x, y, w, h]
            p_xyxy = [p[0], p[1], p[0] + p[2], p[1] + p[3]]

            best_iou = 0.0
            for gt in gt_boxes:
                iou = compute_iou(gt, p_xyxy)
                best_iou = max(best_iou, iou)

            # Assign label: 1 if IoU >= 0.5, 0 otherwise
            label = 1 if best_iou >= iou_threshold else 0

            if label == 0:
                if neg_count >= max_negatives:
                    continue
                neg_count += 1

            dataset.append(
                {
                    "image": img_name,
                    "proposal": p,  # [x, y, w, h]
                    "label": label,
                    "iou": best_iou,
                }
            )
    return dataset


def main() -> None:
    """Run the proposal extraction, evaluation, and preparation pipeline."""
    # --- Configuration ---
    script_dir = Path(__file__).parent
    image_dir = Path("/dtu/datasets1/02516/potholes/images/")
    annotation_dir = Path("/dtu/datasets1/02516/potholes/annotations/")

    # Create output directories if they don't exist
    (script_dir / "figures").mkdir(exist_ok=True)
    (script_dir / "proposals_data").mkdir(exist_ok=True)

    output_plot_path = script_dir / "figures/proposals_evaluation.png"
    train_output_path = script_dir / "proposals_data/train_proposals.pkl"
    val_output_path = script_dir / "proposals_data/val_proposals.pkl"
    test_output_path = script_dir / "proposals_data/test_proposals.pkl"
    selected_k = 1000  # Threshold for proposals
    dataset_iou_threshold = 0.9  # For labeling proposals
    recall_iou_threshold = 0.5  # For evaluating recall
    max_negatives = 50  # Max negatives per image

    # --- 1. Extract Object Proposals ---
    logger.info("Step 1: Extracting object proposals...")
    image_paths = sorted(image_dir.glob("*.png"))
    logger.info(f"Found {len(image_paths)} images.")

    all_proposals = {}

    for img_path in tqdm(image_paths, desc="Processing images"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Run Selective Search
        rects = run_selective_search(img, method="fast")

        # Store proposals
        all_proposals[img_path.name] = rects

    logger.info(f"Processed {len(all_proposals)} images.")

    # --- 2. Load Annotations ---
    logger.info("Step 2: Loading annotations...")
    all_annotations = {}

    for img_name in all_proposals:
        xml_name = Path(img_name).with_suffix(".xml").name
        xml_path = annotation_dir / xml_name

        if xml_path.exists():
            with xml_path.open() as file:
                file_data = file.read()
                dict_data = xmltodict.parse(file_data)

                boxes = []
                if "object" in dict_data["annotation"]:
                    objects = dict_data["annotation"]["object"]
                    if not isinstance(objects, list):
                        objects = [objects]

                    for obj in objects:
                        ymin = int(obj["bndbox"]["ymin"])
                        xmin = int(obj["bndbox"]["xmin"])
                        ymax = int(obj["bndbox"]["ymax"])
                        xmax = int(obj["bndbox"]["xmax"])
                        # Store as [xmin, ymin, xmax, ymax]
                        boxes.append([xmin, ymin, xmax, ymax])
                all_annotations[img_name] = boxes

    logger.info(f"Loaded annotations for {len(all_annotations)} images.")

    # --- 3. Train/Val/Test Split ---
    logger.info("Step 3: Splitting data into Train/Val/Test...")
    image_names = list(all_annotations.keys())
    # First split: 20% Test, 80% Train+Val
    train_val_images, test_images = train_test_split(image_names, test_size=0.2, random_state=42)
    # Second split: 20% of remaining (16% total) for Val, 80% of remaining (64% total) for Train
    train_images, val_images = train_test_split(train_val_images, test_size=0.2, random_state=42)

    logger.info(f"Training set: {len(train_images)} images")
    logger.info(f"Validation set: {len(val_images)} images")
    logger.info(f"Test set: {len(test_images)} images")

    # --- 4. Evaluate Proposals ---
    logger.info("Step 4: Evaluating proposals on training set...")
    k_values = [50, 100, 500, 1000, 1500, 2000]
    recall_results = evaluate_recall(
        train_images, all_proposals, all_annotations, k_values, iou_threshold=recall_iou_threshold
    )
    mabo_results = evaluate_mabo(train_images, all_proposals, all_annotations, k_values)

    logger.info("Recall and MABO at different number of proposals:")
    for k in k_values:
        logger.info(f"Top {k}: Recall={recall_results[k]:.4f}, MABO={mabo_results[k]:.4f}")

    # Save plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_values, [recall_results[k] for k in k_values], marker="o")
    plt.title("Recall vs Number of Proposals")
    plt.xlabel("Number of Proposals")
    plt.ylabel("Recall (IoU >= 0.5)")
    plt.grid(visible=True)

    plt.subplot(1, 2, 2)
    plt.plot(k_values, [mabo_results[k] for k in k_values], marker="o", color="orange")
    plt.title("MABO vs Number of Proposals")
    plt.xlabel("Number of Proposals")
    plt.ylabel("Mean Average Best Overlap")
    plt.grid(visible=True)

    plt.tight_layout()
    plt.savefig(output_plot_path)
    logger.info(f"Evaluation plot saved to {output_plot_path}")

    # --- 5. Prepare Proposals for Training and Testing ---
    logger.info(f"Step 5: Preparing proposals for training and testing (Top {selected_k})...")

    train_proposals_data = prepare_dataset(
        train_images,
        all_proposals,
        all_annotations,
        selected_k,
        dataset_iou_threshold,
        max_negatives,
    )
    logger.info(f"Generated {len(train_proposals_data)} training samples.")
    pos_count = sum(1 for x in train_proposals_data if x["label"] == 1)
    neg_count = len(train_proposals_data) - pos_count
    logger.info(f"Training - Positive samples: {pos_count}")
    logger.info(f"Training - Negative samples: {neg_count}")

    with train_output_path.open("wb") as f:
        pickle.dump(train_proposals_data, f)
    logger.info(f"Saved training proposals to {train_output_path}")

    val_proposals_data = prepare_dataset(
        val_images,
        all_proposals,
        all_annotations,
        selected_k,
        dataset_iou_threshold,
        max_negatives,
    )
    logger.info(f"Generated {len(val_proposals_data)} validation samples.")
    pos_count_val = sum(1 for x in val_proposals_data if x["label"] == 1)
    neg_count_val = len(val_proposals_data) - pos_count_val
    logger.info(f"Validation - Positive samples: {pos_count_val}")
    logger.info(f"Validation - Negative samples: {neg_count_val}")

    with val_output_path.open("wb") as f:
        pickle.dump(val_proposals_data, f)
    logger.info(f"Saved validation proposals to {val_output_path}")

    test_proposals_data = prepare_dataset(
        test_images,
        all_proposals,
        all_annotations,
        selected_k,
        dataset_iou_threshold,
        max_negatives,
    )
    logger.info(f"Generated {len(test_proposals_data)} test samples.")
    pos_count_test = sum(1 for x in test_proposals_data if x["label"] == 1)
    neg_count_test = len(test_proposals_data) - pos_count_test
    logger.info(f"Testing - Positive samples: {pos_count_test}")
    logger.info(f"Testing - Negative samples: {neg_count_test}")

    with test_output_path.open("wb") as f:
        pickle.dump(test_proposals_data, f)
    logger.info(f"Saved test proposals to {test_output_path}")


if __name__ == "__main__":
    main()
