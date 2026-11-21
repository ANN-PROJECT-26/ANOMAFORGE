# P2 - Student CNN, Teacher–Student Feature Matching and integration

## Responsibilities

* Implement a student CNN that learns to match features from a frozen DINOv3 teacher.
* Prepare normal bottle images for training and development, and experiment with image augmentation.
* Train the student using mean squared error between L2-normalized teacher and student features.
* Generate anomaly heatmaps and image-level scores from teacher–student feature differences.

## Files

* `student.py` - Defines the student CNN and its spatial feature output.
* `train\_img\_augmentation.py` - Defines training-image augmentation and preview functions.
* `01\_data\_exploration\_student\_model\_integration\_V1.0.ipynb` - Baseline notebook for dataset exploration, teacher–student training, and anomaly scoring.
* `01\_data\_exploration\_student\_model\_integration\_withaugV2.0.ipynb` - Notebook with an augmentation preview and a single augmented training update.
* `01\_data\_exploration\_student\_model\_integration.py` - Colab-exported Python script for the baseline workflow.

## Evaluation Plan

* Split the 209 normal bottle images into 167 training images and 42 development images.
* Monitor training and development feature-matching losses and save the student checkpoint with the lowest development loss.
* Apply augmentation to training images only; compare teacher and student features from the same input.
* Use feature discrepancies to create anomaly maps and image scores, set a threshold from normal development images, and assess normal and defective test images.

## Status

The baseline notebook contains a completed training run and a single-image anomaly-scoring example. Augmentation is implemented and has been used for a single training update; integration into the full batch-training loop and evaluation of the augmented model are in progress.

The notebooks require the MVTec AD bottle dataset and `P1/teacher.py`. The augmented notebook expects `P2/trn\_img\_augmentation.py`; align that path with the supplied augmentation filename before running it.

