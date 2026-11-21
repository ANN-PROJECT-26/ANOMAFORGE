import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

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


