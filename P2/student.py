# Student CNN for AnomaForge: learns to reproduce features from the frozen
# DINOv3 teacher on normal bottle images. This file defines the network only;
# the notebook creates it, calculates the loss, and trains it.
#
# Input:  [B, 3, 224, 224] - B RGB images after the teacher's preprocessing.
# Output: [B, 384, 14, 14] - a spatial feature map with the same shape as
#         the teacher's output. This is a feature map, not a defect mask.
#
# During training, the notebook compares the student and teacher features.
# The proposal uses MSE on L2-normalized 384-element feature vectors at each
# grid location. The optimizer updates the student; the teacher stays frozen.

# torch.nn provides building blocks such as convolution layers and ReLU.
import torch.nn as nn


# nn.Module is PyTorch's base class for neural network models.
class StudentCNN(nn.Module):
    def __init__(self):
        # Initialize nn.Module so PyTorch can keep track of our layers and
        # their trainable weights and biases.
        super().__init__()

        # Sequential applies these layers in the order they are listed.
        #
        # Conv2d(in_channels, out_channels, kernel_size, stride, padding):
        # - in_channels/out_channels: number of input/output feature maps.
        # - kernel_size=3: each filter is 3 x 3 pixels or feature locations.
        # - stride=2: move the filter by two locations, reducing spatial size.
        # - padding=1: add a one-pixel zero border before the convolution.
        #
        # ReLU introduces a nonlinearity by replacing negative values with 0.
        # The convolution filters and biases are learned during training.
        self.network = nn.Sequential(
            # B RGB images: [B, 3, 224, 224] -> [B, 32, 112, 112].
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # [B, 32, 112, 112] -> [B, 64, 56, 56].
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # [B, 64, 56, 56] -> [B, 128, 28, 28].
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # [B, 128, 28, 28] -> [B, 256, 14, 14].
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # Stride 1 keeps the 14 x 14 grid; 384 output channels match
            # the teacher's feature-vector dimension at each grid location.
            # There is no final ReLU, so output features may be negative.
            # [B, 256, 14, 14] -> [B, 384, 14, 14].
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, images):
        # Called automatically when the notebook runs student_model(images).
        # Pass the image batch through all layers and return its feature map.
        return self.network(images)