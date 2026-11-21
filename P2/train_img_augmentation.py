import random
from PIL import Image, ImageOps, ImageEnhance

AUGMENT_CONFIG = {
    'enabled': True,
    'seed': 42,
    'horizontal_flip_probability': 0.5,
    'rotation_probability': 0.5,
    'max_rotation_degrees': 10.0,
    'brightness_probability': 0.5,
    'brightness_min': 0.95,
    'brightness_max': 1.05,
}

# A dedicated generator controls augmentation of training images only.
augmentation_rng = random.Random(AUGMENT_CONFIG['seed'])

def augment_normal_image(image, rng):
    """Return a randomly transformed RGB PIL image, without changing input."""
    result = image.convert('RGB')
    cfg = AUGMENT_CONFIG
    if not cfg['enabled']:
        return result

    if rng.random() < cfg['horizontal_flip_probability']:
        result = ImageOps.mirror(result)

    if rng.random() < cfg['rotation_probability']:
        angle = rng.uniform(-cfg['max_rotation_degrees'], cfg['max_rotation_degrees'])
        result = result.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(255, 255, 255)
        )

    if rng.random() < cfg['brightness_probability']:
        factor = rng.uniform(cfg['brightness_min'], cfg['brightness_max'])
        result = ImageEnhance.Brightness(result).enhance(factor)

    return result


def preview_augmentation(image_path):
    """Display examples for one training image without advancing the training RNG."""
    from pathlib import Path
    import matplotlib.pyplot as plt

    # Preview a training image and examples of each transformation.
    # The preview RNG is independent of the training augmentation RNG.
    with Image.open(image_path) as image:
        preview_original = image.convert('RGB')
    preview_rng = random.Random(123)

    preview_examples = [
        ('Original', preview_original),
        ('Horizontal flip', ImageOps.mirror(preview_original)),
        ('Rotation +10 degrees', preview_original.rotate(
            10, resample=Image.Resampling.BILINEAR, expand=False,
            fillcolor=(255, 255, 255))),
        ('Brightness x1.05', ImageEnhance.Brightness(preview_original).enhance(1.05)),
        ('Random example 1', augment_normal_image(preview_original, preview_rng)),
        ('Random example 2', augment_normal_image(preview_original, preview_rng)),
        ('Random example 3', augment_normal_image(preview_original, preview_rng)),
        ('Random example 4', augment_normal_image(preview_original, preview_rng)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for axis, (title, image) in zip(axes.flat, preview_examples):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis('off')
    plt.tight_layout()
    plt.show()
    print('Training example:', Path(image_path).name)
    print('Augmentation settings:', AUGMENT_CONFIG)
