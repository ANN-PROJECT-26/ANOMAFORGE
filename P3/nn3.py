"""P3: NN3 segmentation CNN, training, diagnostics and inference for AnomaForge.

NN3 input : [B, 4, 224, 224] = RGB (0..1) + scaled teacher-student discrepancy map
NN3 output: [B, 1, 224, 224] segmentation logits, so then apply sigmoid for probabilities

Trained only on synthetic defects from p3_synthetic.py, real test defects and
masks are never used for training, thresholds, or checkpoint selection
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from p3_synthetic import SIZE, discrepancy_maps, load_normal


# -------------------------------------------------------------------- model
def _block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class NN3(nn.Module):
    """Small U-Net. Full-resolution skip connections carry the raw RGB detail
    forward, which is the "RGB skip" idea in the proposal diagram."""

    def __init__(self, in_channels=4, base=16):
        super().__init__()
        b = base
        self.enc1 = _block(in_channels, b) # 224
        self.enc2 = _block(b, 2 * b)  # 112
        self.enc3 = _block(2 * b, 4 * b) # 56
        self.bott = _block(4 * b, 8 * b) # 28
        self.up3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, stride=2)
        self.dec3 = _block(8 * b, 4 * b)
        self.up2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2)
        self.dec2 = _block(4 * b, 2 * b)
        self.up1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2)
        self.dec1 = _block(2 * b, b)
        self.head = nn.Conv2d(b, 1, 1) # logits, no sigmoid
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.bott(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(m), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


# Computes a differentiable, batch-wide Soft Dice Loss (1 - Dice) that
# directly optimizes overlap between predicted probabilities and ground-truth masks
def dice_loss(logits, target, eps=1.0):
    """Soft Dice over the whole batch (stable when some masks are all-zero)."""
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    return 1 - (2 * inter + eps) / (p.sum() + target.sum() + eps)

# Inspects the training dataset's pixel balance once up front and returns a configured loss_fn(logits, target)
def make_loss(train_ds, use_dice=True, max_pos_weight=30.0):
    """BCE-with-logits (+ Dice). pos_weight offsets tiny defect areas."""

    # computes the exact fraction of defect pixels across the entire training set
    pos_frac = max(float(train_ds.masks.float().mean()), 1e-6)
    pos_weight = min((1 - pos_frac) / pos_frac, max_pos_weight)

    def loss_fn(logits, target):
        w = torch.tensor(pos_weight, device=logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=w)
        if use_dice:
            loss = loss + dice_loss(logits, target)
        return loss

    loss_fn.pos_weight = pos_weight
    return loss_fn


# model evaluation - validation/thresholding pipeline for the nn3 model

# Runs inference over an entire dataset and collects the predicted probability maps alongside ground-truth masks on the CPU
@torch.no_grad()
def predict_probs(model, ds, device, batch_size=16):
    """Return (probs [N,H,W] fp16, masks [N,H,W] bool) for a dataset (CPU)."""
    model.eval()
    probs, masks = [], []
    for x, y in DataLoader(ds, batch_size=batch_size, shuffle=False):
        probs.append(torch.sigmoid(model(x.to(device))).cpu()[:, 0].half()) # convert logits to probabilities
        masks.append(y[:, 0] > 0.5) # binarize the ground-truth mask into a bool tensor
    return torch.cat(probs), torch.cat(masks)

# Computes per-image Dice Similarity Coefficient (F1) and Intersection over Union for binary predictions
def dice_iou(pred, mask, eps=1e-6):
    """Per-sample Dice and IoU for boolean tensors [N,H,W]."""
    inter = (pred & mask).flatten(1).sum(1).float() # Counts pixels where both pred and mask are True
    p, m = pred.flatten(1).sum(1).float(), mask.flatten(1).sum(1).float()
    union = p + m - inter # Counts pixels where either pred or mask is True
    return (2 * inter + eps) / (p + m + eps), (inter + eps) / (union + eps) # add epsilon for smoothing, set to 1e-6


# Evaluates segmentation quality at a specific probability cutoff while accounting for defect-free images
def segmentation_metrics(probs, masks, threshold=0.5):
    """Dice/IoU on defective samples; false-positive pixel rate on clean ones."""
    pred = probs.float() > threshold
    dice, iou = dice_iou(pred, masks)
    defective = masks.flatten(1).any(1) # separate the images which contain a ground truth defect
    
    # Averages the Dice and IoU scores exclusively across images that actually have defects
    out = {
        "dice": float(dice[defective].mean()) if defective.any() else float("nan"),
        "iou": float(iou[defective].mean()) if defective.any() else float("nan"),
        "n_defective": int(defective.sum()),
    }
    if (~defective).any():
        # compute fraction of pixels falsely flagged as defective across all clean images
        out["clean_fp_pixel_rate"] = float(pred[~defective].float().mean())
    return out

# Tunes the optimal probability threshold t on a validation/dev set to maximize the Dice score
def pick_threshold(probs, masks, grid=None):
    """Segmentation threshold chosen on SYNTHETIC DEV masks only."""
    grid = np.linspace(0.1, 0.9, 17) if grid is None else grid
    scores = [segmentation_metrics(probs, masks, t)["dice"] for t in grid]
    return float(grid[int(np.nanargmax(scores))]), dict(zip(map(float, grid), scores))


# NN3 Training
def train_nn3(train_ds, dev_ds, checkpoint_path, epochs=30, batch_size=16,
              lr=1e-3, base=16, use_dice=True, device=None, seed=42):
    """Train NN3 on synthetic data; keep the checkpoint with lowest dev loss.

    Returns (model, history) with the best checkpoint already loaded.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    model = NN3(base=base).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = make_loss(train_ds, use_dice=use_dice)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)

    def run(loader, training):
        model.train(training)
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.set_grad_enabled(training):
                loss = loss_fn(model(x), y)
                if not torch.isfinite(loss):
                    raise ValueError("Loss is not finite.")
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            total += loss.item() * len(x)
            n += len(x)
        return total / n

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "dev_loss": [], "dev_dice": []}
    best = math.inf
    for epoch in range(1, epochs + 1):
        tr = run(train_loader, True)
        dv = run(dev_loader, False)
        probs, masks = predict_probs(model, dev_ds, device, batch_size)
        dice = segmentation_metrics(probs, masks, 0.5)["dice"]
        history["train_loss"].append(tr)
        history["dev_loss"].append(dv)
        history["dev_dice"].append(dice)
        print(f"Epoch {epoch:02d}/{epochs} | train {tr:.4f} | dev {dv:.4f} | dev Dice@0.5 {dice:.3f}")
        if dv < best:
            best = dv
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "dev_loss": dv, "base": base, "map_scale": train_ds.map_scale,
                        "pos_weight": loss_fn.pos_weight}, checkpoint_path)
            print("  Saved best NN3 checkpoint.")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Loaded best epoch:", ckpt["epoch"], "dev loss:", ckpt["dev_loss"])
    return model, history


# Run diagnostics on NN3
def plot_history(history, out_file=None):
    import matplotlib.pyplot as plt
    ep = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, history["train_loss"], label="Training")
    ax[0].plot(ep, history["dev_loss"], label="Development")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].legend(); ax[0].grid(True)
    ax[1].plot(ep, history["dev_dice"])
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Synthetic dev Dice @0.5"); ax[1].grid(True)
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file, dpi=100)
    plt.show()


def plot_overlays(ds, probs, indices, threshold=0.5, out_file=None):
    """RGB | discrepancy | true mask | predicted probability | thresholded."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(indices), 5, figsize=(15, 3 * len(indices)))
    ax = np.atleast_2d(ax)
    for r, i in enumerate(indices):
        x, y = ds[i]
        panels = [(x[:3].permute(1, 2, 0), None, f"{ds.kinds[i]}"),
                  (x[3], "inferno", "discrepancy"),
                  (y[0], "gray", "true mask"),
                  (probs[i].float(), "viridis", "NN3 prob"),
                  (probs[i].float() > threshold, "gray", f"pred @{threshold:.2f}")]
        for c, (img, cmap, title) in enumerate(panels):
            ax[r, c].imshow(img, cmap=cmap); ax[r, c].set_title(title); ax[r, c].axis("off")
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file, dpi=100)
    plt.show()


# Inference function - reduces a pixel-level segmentation score map into a single image-level classification
# score by averaging the top 1% highest-scoring pixels
def top_percent_score(score_map, fraction=0.01):
    """Image score = mean of the highest-scoring 1% of pixels (same rule as the
    baseline, so the NN3-vs-baseline comparison is controlled)."""
    flat = score_map.flatten()
    k = max(1, math.ceil(fraction * flat.numel()))
    return torch.topk(flat, k).values.mean().item()


class NN3Refiner:
    """Should be called in P4; wraps frozen teacher + student + trained NN3

    refiner = NN3Refiner(teacher, student_model, "path/to/nn3_best.pt")
    out = refiner.refine(pil_image)
      out["baseline_map"], out["baseline_score"]   # NN1+NN2 cosine discrepancy
      out["prob_map"],     out["nn3_score"]        # NN3 refined [224,224], [0,1]
      out["mask"]  (needs seg_threshold)           # bool [224,224]
    """

    def __init__(self, teacher, student, checkpoint_path, seg_threshold=0.5, device=None):
        self.teacher, self.student = teacher, student
        self.device = torch.device(device or teacher.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.map_scale = ckpt["map_scale"]  # must match training
        self.model = NN3(base=ckpt["base"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.seg_threshold = seg_threshold

    @torch.no_grad()
    def refine(self, image):
        # Same preprocessing as the training data in synthetic.py; resize to 224x224, uint8 round trip.
        rgb = load_normal_from_pil(image)
        pil = Image.fromarray((rgb * 255).round().astype(np.uint8))
        dmap = discrepancy_maps(self.teacher, self.student, [pil])[0]
        rgb_t = torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float() / 255.0
        x = torch.cat([rgb_t, (dmap * self.map_scale).unsqueeze(0)], 0).unsqueeze(0)
        prob = torch.sigmoid(self.model(x.to(self.device)))[0, 0].cpu()
        return {
            "baseline_map": dmap, "baseline_score": top_percent_score(dmap),
            "prob_map": prob, "nn3_score": top_percent_score(prob),
            "mask": prob > self.seg_threshold,
        }


def load_normal_from_pil(pil_image):
    """Same resize as synthetic.load_normal, but from a PIL image"""
    im = pil_image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0