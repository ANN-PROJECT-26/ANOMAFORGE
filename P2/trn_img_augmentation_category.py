"""Category-aware, on-the-fly augmentation for MVTec normal training images.

The policies below are conservative starting points, not validated guarantees.
Review each category and keep only transformations that are normal for your
inspection setup. Development and test images should not be augmented here.
"""

import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


def policy(flip=0.0, rotation=0.0, degrees=0.0, brightness=0.2,
           brightness_min=0.97, brightness_max=1.03):
    """Build one editable set of probabilities and transformation limits."""
    return {
        'horizontal_flip_probability': flip,
        'rotation_probability': rotation,
        'max_rotation_degrees': degrees,
        'brightness_probability': brightness,
        'brightness_min': brightness_min,
        'brightness_max': brightness_max,
    }


AUGMENT_CONFIG = {
    'enabled': True,
    'seed': 42,
    'categories': {
        # Roughly symmetric / variable-pose objects: mild geometric changes.
        'bottle':     policy(flip=0.5, rotation=0.35, degrees=7),
        'hazelnut':   policy(flip=0.35, rotation=0.35, degrees=8),
        'metal_nut':  policy(rotation=0.35, degrees=8),
        'screw':      policy(rotation=0.25, degrees=6),

        # Textures: occasional mirror; avoid rotation's artificial corners.
        'carpet':     policy(flip=0.35),
        'leather':    policy(flip=0.35),
        'tile':       policy(flip=0.35),
        # Preserve the direction of grid and wood until validated.
        'grid':       policy(),
        'wood':       policy(),

        # Preserve object orientation / placement and fine markings.
        'cable':      policy(),
        'capsule':    policy(),
        'pill':       policy(),
        'toothbrush': policy(),
        'transistor': policy(),
        'zipper':     policy(),
    },
}

# Kept for interactive use; the notebook supplies its own per-epoch RNG.
augmentation_rng = random.Random(AUGMENT_CONFIG['seed'])


def category_from_path(image_path):
    """Read the category from .../<category>/train/good/<image>."""
    path = Path(image_path)
    if path.parent.name != 'good' or path.parent.parent.name != 'train':
        raise ValueError(f'Expected <category>/train/good/<image>: {path}')
    category = path.parent.parent.parent.name
    if category not in AUGMENT_CONFIG['categories']:
        raise ValueError(f'No augmentation policy for category: {category}')
    return category


def augment_normal_image(image, rng, category=None):
    """Return an RGB view using this category's independently sampled transforms.

    category must be passed when augmentation is enabled; this prevents older
    notebook callers from silently using the wrong category policy.
    """
    result = image.convert('RGB')
    if not AUGMENT_CONFIG['enabled']:
        return result
    if category not in AUGMENT_CONFIG['categories']:
        raise ValueError(
            f'Unknown or missing augmentation category: {category!r}. '
            'Pass category=category_from_path(image_path).'
        )

    cfg = AUGMENT_CONFIG['categories'][category]
    if rng.random() < cfg['horizontal_flip_probability']:
        result = ImageOps.mirror(result)

    if rng.random() < cfg['rotation_probability']:
        angle = rng.uniform(-cfg['max_rotation_degrees'], cfg['max_rotation_degrees'])
        result = result.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(255, 255, 255),
        )

    if rng.random() < cfg['brightness_probability']:
        factor = rng.uniform(cfg['brightness_min'], cfg['brightness_max'])
        result = ImageEnhance.Brightness(result).enhance(factor)

    return result


def preview_augmentation(image_path, category=None):
    """Show random views of one normal image without advancing training RNG."""
    import matplotlib.pyplot as plt

    category = category or category_from_path(image_path)
    if category not in AUGMENT_CONFIG['categories']:
        raise ValueError(f'No augmentation policy for category: {category}')
    with Image.open(image_path) as image:
        original = image.convert('RGB')

    preview_rng = random.Random(123)
    examples = [('Original', original)]
    for number in range(1, 8):
        view = augment_normal_image(original, preview_rng, category=category)
        examples.append((f'Random view {number}', view))

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for axis, (title, image) in zip(axes.flat, examples):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis('off')
    fig.suptitle(f'{category}: augmentation preview')
    plt.tight_layout()
    plt.show()
    print('Category:', category, '| Training example:', Path(image_path).name)
    print('Settings:', AUGMENT_CONFIG['categories'][category])
