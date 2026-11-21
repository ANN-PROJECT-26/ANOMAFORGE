import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve


def compute_image_score(anomaly_map):
    """
    Compute the image-level anomaly score using
    the mean of the highest-scoring 1% of pixels.

    Input:
        anomaly_map: 2D array [H, W]

    Output:
        Single anomaly score (float)
    """
    scores = np.asarray(anomaly_map, dtype=np.float64)

    if scores.ndim != 2:
        raise ValueError("Expected a 2D anomaly map [H, W]")

    # Flatten the anomaly map
    flat_scores = scores.flatten()

    # Select top 1% pixels
    k = max(1, int(np.ceil(len(flat_scores) * 0.01)))
    top_scores = np.sort(flat_scores)[-k:]

    return float(np.mean(top_scores))


def compute_feature_discrepancy(teacher_features, student_features):
    """
    Compute cosine discrepancy between teacher
    and student spatial features.

    Input:
        teacher_features: [B, C, H, W]
        student_features: [B, C, H, W]

    Output:
        anomaly_map: [B, 1, 224, 224]
    """

    if teacher_features.shape != student_features.shape:
        raise ValueError("Feature shapes must match")

    # Compute cosine similarity across channels
    similarity = F.cosine_similarity(
        teacher_features,
        student_features,
        dim=1
    )

    # Convert similarity to discrepancy
    discrepancy = 1.0 - similarity

    # Add channel dimension
    discrepancy = discrepancy.unsqueeze(1)

    # Resize to original model input size
    anomaly_map = F.interpolate(
        discrepancy,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    return anomaly_map


def compute_image_auroc(image_labels, image_scores):
    """
    Compute image-level AUROC.

    Input:
        image_labels: [N], 0 = normal, 1 = defective
        image_scores: [N], image-level anomaly scores

    Output:
        Image AUROC (float)
    """
    labels = np.asarray(image_labels).reshape(-1)
    scores = np.asarray(image_scores).reshape(-1)

    if labels.shape != scores.shape:
        raise ValueError("Labels and scores must match")

    if np.unique(labels).size != 2:
        raise ValueError("AUROC requires both classes")

    return float(roc_auc_score(labels, scores))


def compute_pixel_auroc(ground_truth_masks, anomaly_maps):
    """
    Compute pixel-level AUROC across all images.

    Input:
        ground_truth_masks: [N, H, W], binary masks
        anomaly_maps: [N, H, W], continuous scores

    Output:
        Pixel AUROC (float)
    """
    masks = np.asarray(ground_truth_masks)
    scores = np.asarray(anomaly_maps)

    if masks.shape != scores.shape:
        raise ValueError("Masks and anomaly maps must match")

    # Flatten all pixels across all images
    labels = (masks > 0).astype(np.uint8).flatten()
    scores = scores.flatten()

    if np.unique(labels).size != 2:
        raise ValueError("AUROC requires both classes")

    return float(roc_auc_score(labels, scores))

def compute_segmentation_metrics(
    ground_truth_masks, anomaly_maps, threshold
):
    """
    Compute pixel-level Dice, IoU, Precision and Recall.

    Input:
        ground_truth_masks: Binary masks [N, H, W]
        anomaly_maps: Continuous scores [N, H, W]
        threshold: Segmentation threshold

    Output:
        Dictionary of segmentation metrics
    """

    masks = np.asarray(ground_truth_masks)
    scores = np.asarray(anomaly_maps)

    if masks.shape != scores.shape:
        raise ValueError("Masks and anomaly maps must match")

    # Convert to binary masks
    gt = masks > 0
    pred = scores >= threshold

    # Compute confusion counts across all pixels
    tp = int(np.sum(gt & pred))
    fp = int(np.sum(~gt & pred))
    fn = int(np.sum(gt & ~pred))
    tn = int(np.sum(~gt & ~pred))

    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator > 0 else 0.0

    dice = safe_divide(2 * tp, 2 * tp + fp + fn)
    iou = safe_divide(tp, tp + fp + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }


def compute_image_threshold(normal_dev_scores):
    """
    Set the image threshold using the 95th percentile
    of normal development image scores.
    """
    scores = np.asarray(normal_dev_scores, dtype=np.float64)

    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Invalid development scores")

    return float(np.percentile(scores, 95))


def select_segmentation_threshold(dev_masks, dev_maps):
    """
    Select the threshold that maximizes pixel-level Dice
    using synthetic development data only.
    """
    masks = np.asarray(dev_masks)
    scores = np.asarray(dev_maps)

    if masks.shape != scores.shape:
        raise ValueError("Masks and anomaly maps must match")

    labels = (masks > 0).astype(np.uint8).flatten()
    scores = scores.flatten()

    if np.unique(labels).size != 2:
        raise ValueError("Both pixel classes are required")

    precision, recall, thresholds = precision_recall_curve(
        labels, scores
    )

    # Calculate Dice for each threshold
    p = precision[:-1]
    r = recall[:-1]

    dice = np.divide(
        2 * p * r,
        p + r,
        out=np.zeros_like(p),
        where=(p + r) > 0
    )

    best_index = np.argmax(dice)

    return float(thresholds[best_index])


def inspection_decision(image_score, image_threshold):
    """
    Generate a PASS or REVIEW recommendation.
    """
    if image_score > image_threshold:
        return "REVIEW"

    return "PASS"


def calibrate_thresholds(
    normal_dev_maps, synthetic_dev_masks, synthetic_dev_maps
):
    """
    Calibrate thresholds using development data only.
    All maps and masks have shape [N, H, W].
    """

    # Compute normal development image scores
    normal_dev_scores = [
        compute_image_score(m) for m in normal_dev_maps
    ]

    image_threshold = compute_image_threshold(
        normal_dev_scores
    )

    segmentation_threshold = select_segmentation_threshold(
        synthetic_dev_masks,
        synthetic_dev_maps
    )

    return {
        "image_threshold": image_threshold,
        "segmentation_threshold": segmentation_threshold
    }


def evaluate_model(
    image_labels, ground_truth_masks, anomaly_maps, thresholds
):
    """
    Evaluate a model using previously selected thresholds.

    Input:
        image_labels: [N]
        ground_truth_masks: [N, H, W]
        anomaly_maps: [N, H, W]
        thresholds: Calibrated threshold dictionary

    Output:
        Evaluation metrics and inspection decisions
    """

    anomaly_maps = np.asarray(anomaly_maps)
    ground_truth_masks = np.asarray(ground_truth_masks)
    image_labels = np.asarray(image_labels).reshape(-1)

    if anomaly_maps.ndim != 3:
        raise ValueError("Expected anomaly maps [N, H, W]")

    if anomaly_maps.shape != ground_truth_masks.shape:
        raise ValueError("Masks and maps must match")

    if len(image_labels) != len(anomaly_maps):
        raise ValueError("Number of labels and maps must match")

    # Image-level anomaly scores
    image_scores = np.array([
        compute_image_score(m) for m in anomaly_maps
    ])

    # Primary evaluation metrics
    image_auc = compute_image_auroc(
        image_labels, image_scores
    )

    pixel_auc = compute_pixel_auroc(
        ground_truth_masks, anomaly_maps
    )

    # Additional segmentation metrics
    segmentation_results = compute_segmentation_metrics(
        ground_truth_masks,
        anomaly_maps,
        thresholds["segmentation_threshold"]
    )

    # Inspection recommendations
    decisions = [
        inspection_decision(
            score, thresholds["image_threshold"]
        )
        for score in image_scores
    ]

    return {
        "image_auroc": image_auc,
        "pixel_auroc": pixel_auc,
        **segmentation_results,
        "image_scores": image_scores,
        "decisions": decisions
    }
