import numpy as np


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
