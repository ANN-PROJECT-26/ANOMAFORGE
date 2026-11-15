"""P3: synthetic defects + 4-channel dataset for NN3 (AnomaForge)

Pipeline per sample
    normal bottle -> resize 224x224 -> synthetic defect + binary mask
    -> frozen teacher/student -> cosine discrepancy map (224x224)
    -> x = [R, G, B in 0..1, scaled map]  (4 channels),  y = mask (1 channel)

Teacher and student are both frozen, so samples are generated once and cached
DELETE THE CACHE FILE if the student checkpoint changes
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset

SIZE = 224
KINDS = ("scratch", "stain", "cutpaste")


# ----------------------------------------------------------------- split / io


# Usage: arguments are the saved best student checkpoint path from P2, path to Bottle directory
def load_split(checkpoint_path, bottle_dir):
    # Reuse P2's split from the saved P2 student checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    good = Path(bottle_dir) / "train" / "good"
    # get all paths to training and devlopment files
    training = [good / n for n in ckpt["train_files"]]
    dev = [good / n for n in ckpt["dev_files"]]
    assert set(training).isdisjoint(dev) # these should not overlap
    return training, dev # returns lists of file paths to training and development files

# Convert image from path into a 224x224 Numpy array of float32's
def load_normal(path):
    # Plain bilinear resize (no crop), which is what teacher.py asks the processor to do, so
    # image/mask geometry matches the teacher's input
    with Image.open(path) as im:
        im = im.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


# ------------------------------------------------------------------ defects

# generate artificial scratch
def _scratch(img, rng):
    mask_im = Image.new("L", (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(mask_im)
    x, y = rng.uniform(30, SIZE - 30, 2)
    angle = rng.uniform(0, 2 * np.pi)
    width = int(rng.integers(1, 4))
    for _ in range(int(rng.integers(3, 7))):
        angle += rng.normal(0, 0.35)
        step = rng.uniform(8, 25)
        nx = np.clip(x + step * np.cos(angle), 0, SIZE - 1)
        ny = np.clip(y + step * np.sin(angle), 0, SIZE - 1)
        draw.line([(x, y), (nx, ny)], fill=255, width=width)
        x, y = nx, ny
    mask = np.asarray(mask_im) > 0
    soft = np.asarray(mask_im.filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255.0
    soft = np.clip(soft * 1.5, 0, 1)[..., None]
    color = rng.choice([0.08, 0.92]) * np.ones(3, np.float32)
    alpha = rng.uniform(0.6, 0.9)
    return img * (1 - alpha * soft) + color * alpha * soft, mask

# generate artificial stain
def _stain(img, rng):
    blob = Image.new("L", (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(blob)
    cx, cy = rng.uniform(40, SIZE - 40, 2)
    for _ in range(int(rng.integers(2, 5))):
        r = rng.uniform(6, 22)
        ox, oy = rng.normal(0, 8, 2)
        draw.ellipse([cx + ox - r, cy + oy - r, cx + ox + r, cy + oy + r], fill=255)
    blob = blob.filter(ImageFilter.GaussianBlur(rng.uniform(2.5, 6)))
    soft = np.asarray(blob, np.float32)
    soft = (soft / max(soft.max(), 1e-6))[..., None]
    color = rng.uniform(0.0, 0.5, 3).astype(np.float32)
    alpha = rng.uniform(0.5, 0.85)
    return img * (1 - alpha * soft) + color * alpha * soft, soft[..., 0] > 0.3

# generate cut-paste region
def _cutpaste(img, rng):
    h, w = (int(v) for v in rng.integers(14, 56, 2))
    for _ in range(20):  # find a destination that doesn't overlap the source
        sy, sx = int(rng.integers(0, SIZE - h)), int(rng.integers(0, SIZE - w))
        dy, dx = int(rng.integers(0, SIZE - h)), int(rng.integers(0, SIZE - w))
        if abs(sy - dy) > h or abs(sx - dx) > w:
            break
    patch = img[sy:sy + h, sx:sx + w].copy()
    if rng.random() < 0.5:
        patch = patch[:, ::-1]
    patch = np.clip(patch * rng.uniform(0.8, 1.2), 0, 1)
    out = img.copy()
    out[dy:dy + h, dx:dx + w] = patch
    mask = np.zeros((SIZE, SIZE), bool)
    mask[dy:dy + h, dx:dx + w] = True
    return out, mask


_FUNCS = {"scratch": _scratch, "stain": _stain, "cutpaste": _cutpaste}


# helper method that calls one of the three above methods to create a synthetic defect
def make_defect(img, rng, kind=None, min_pixels=20):
    """Return (defective_img float32 HxWx3, mask bool HxW, kind).

    Retries if the defect is invisible/empty so labels are never all-zero
    for a "defective" sample.
    """
    kind = kind or str(rng.choice(KINDS))
    for _ in range(10):
        out, mask = _FUNCS[kind](img, rng)
        if mask.sum() >= min_pixels:
            return np.clip(out, 0, 1).astype(np.float32), mask, kind
    return np.clip(out, 0, 1).astype(np.float32), mask, kind


# -------------------------------------------------------- discrepancy maps
@torch.no_grad()
def discrepancy_maps(teacher, student, pil_images):
    """Batched version of the notebook's anomaly map: [B, 224, 224] on CPU."""
    student.eval()
    pixels, t_feat = teacher.extract(pil_images)
    s_feat = student(pixels)
    patch_map = 1 - F.cosine_similarity(t_feat, s_feat, dim=1)  # [B,14,14]
    full = F.interpolate(patch_map.unsqueeze(1), size=(SIZE, SIZE),
                         mode="bilinear", align_corners=False)
    return full[:, 0].cpu()


# This class is used to generate a dataset of synthetic defects and masks
class SyntheticDefectDataset(Dataset):
    """Cached (4-channel input, mask) pairs for NN3.

    paths:          normal-image paths (train split OR dev split, never mixed)
    per_image:      synthetic samples generated per original
    clean_fraction: share of samples with NO defect (all-zero mask)
    map_scale:      multiplier on the raw discrepancy so channel 4 is ~O(1)
    augment:        random flips applied jointly to image, map and mask
    """

    def __init__(self, paths, teacher, student, per_image=8, clean_fraction=0.2,
                 seed=0, batch_size=16, cache_file=None, augment=False,
                 map_scale=10.0):
        self.augment = augment
        self.map_scale = map_scale
        cache_file = Path(cache_file) if cache_file else None

        if cache_file is not None and cache_file.exists():
            d = torch.load(cache_file, weights_only=False)
            self.rgb, self.maps, self.masks, self.kinds = (
                d["rgb"], d["maps"], d["masks"], d["kinds"])
            return

        rgb, masks, kinds = [], [], []
        for i, p in enumerate(paths):
            base = load_normal(p)
            # create per_image synthetic defects of each (chosen) base image
            for k in range(per_image):
                rng = np.random.default_rng([seed, i, k])
                # leave a specified portion of the images clean
                if rng.random() < clean_fraction:
                    img, m, kind = base, np.zeros((SIZE, SIZE), bool), "clean"
                else:
                    img, m, kind = make_defect(base, rng)
                rgb.append((img * 255).round().astype(np.uint8))
                masks.append(m.astype(np.uint8)) # record mask used to create defect (0's if none)
                kinds.append(kind) # scratch, stain, or cut-paste

        # The teacher sees exactly the uint8 image that NN3 will see
        maps = []
        for s in range(0, len(rgb), batch_size):
            pil = [Image.fromarray(a) for a in rgb[s:s + batch_size]]
            maps.append(discrepancy_maps(teacher, student, pil).half())

        self.rgb = torch.from_numpy(np.stack(rgb))        # [N,224,224,3] uint8
        self.maps = torch.cat(maps)                        # [N,224,224] fp16
        self.masks = torch.from_numpy(np.stack(masks))     # [N,224,224] uint8
        self.kinds = kinds
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"rgb": self.rgb, "maps": self.maps,
                        "masks": self.masks, "kinds": self.kinds}, cache_file)

    def __len__(self):
        return len(self.kinds)

    def __getitem__(self, i):
        rgb = self.rgb[i].permute(2, 0, 1).float() / 255.0          # [3,H,W]
        dmap = self.maps[i].float().unsqueeze(0) * self.map_scale   # [1,H,W]
        x = torch.cat([rgb, dmap], dim=0)                           # [4,H,W]
        y = self.masks[i].float().unsqueeze(0)                      # [1,H,W]
        if self.augment:
            if torch.rand(1) < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if torch.rand(1) < 0.5:
                x, y = x.flip(-2), y.flip(-2)
        return x, y


def preview(ds, indices, out_file="p3_preview.png"):
    """Save a grid: RGB | discrepancy map | mask for a few samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(indices), 3, figsize=(9, 3 * len(indices)))
    ax = np.atleast_2d(ax)
    for r, i in enumerate(indices):
        x, y = ds[i]
        ax[r, 0].imshow(x[:3].permute(1, 2, 0)); ax[r, 0].set_title(ds.kinds[i])
        ax[r, 1].imshow(x[3], cmap="inferno"); ax[r, 1].set_title("discrepancy")
        ax[r, 2].imshow(y[0], cmap="gray"); ax[r, 2].set_title("mask")
        for a in ax[r]:
            a.axis("off")
    plt.tight_layout(); plt.savefig(out_file, dpi=100); plt.close(fig)