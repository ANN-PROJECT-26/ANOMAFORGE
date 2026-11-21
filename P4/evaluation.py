import numpy as np
import torch
import torch.nn.functional as F

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

