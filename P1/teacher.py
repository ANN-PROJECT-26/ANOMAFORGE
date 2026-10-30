"""Frozen DINOv3 feature extractor for AnomaForge (Stage 2).

The teacher is pretrained by Meta. This module does NOT train DINOv3.

Input:  PIL RGB image (or a list of PIL RGB images).
Output: (preprocessed_pixels, spatial_features)
        preprocessed_pixels: [B, 3, 224, 224]
        spatial_features:   [B, 384, 14, 14] for DINOv3 ViT-S/16.

Both outputs use the model's own image preprocessing. The pixel tensor can
later be used as the input to our student CNN, avoiding mismatched normalization.

Model: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
"""

import torch
from transformers import AutoImageProcessor, AutoModel


MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"


class DINOv3Teacher:
    """Load a pretrained, frozen DINOv3 ViT-S/16 and extract patch features."""

    def __init__(self, model_id=MODEL_ID, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def extract(self, images):
        """Return normalized pixels and a spatial patch-feature map.

        Pass a PIL RGB image (or list of PIL images), NOT the [0,1] tensor
        created in the Stage 1 exploration notebook. This function applies
        DINOv3's own image processor exactly once.
        """
        batch = self.processor(
            images=images,
            return_tensors="pt",
            size={"height": 224, "width": 224},
        )
        pixel_values = batch["pixel_values"].to(self.device)
        outputs = self.model(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state

        config = self.model.config
        patch_size = config.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_height, patch_width = patch_size
        else:
            patch_height = patch_width = patch_size

        batch_size, _, height, width = pixel_values.shape
        grid_height = height // patch_height
        grid_width = width // patch_width
        prefix_count = 1 + int(getattr(config, "num_register_tokens", 0))
        patch_tokens = tokens[:, prefix_count:, :]

        expected_patches = grid_height * grid_width
        if patch_tokens.shape[1] != expected_patches:
            raise ValueError(
                f"Expected {expected_patches} spatial patches; "
                f"got {patch_tokens.shape[1]}. Check preprocessing or model config."
            )

        feature_map = patch_tokens.transpose(1, 2).reshape(
            batch_size, patch_tokens.shape[-1], grid_height, grid_width
        )
        return pixel_values, feature_map
