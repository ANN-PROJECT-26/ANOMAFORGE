AnomaForge — Stage 1: Dataset Exploration



- `P1/`, `P2/`, `P3/`, `P4/`: empty individual workspaces (with `.gitkeep` so GitHub preserves them). 
- `final/01_data_exploration.ipynb`: our only notebook for Stage 1.
- `.gitignore`: protects the public repository from large datasets, model weights, and accidental credentials.


1. Obtain the MVTec AD dataset from the original source: https://www.mvtec.com/research-teaching/datasets/mvtec-ad (follow the provider's access and license instructions).
2. Unpack the `bottle` category into Google Drive, for example:

   ```text
   MyDrive/AnomaForge/data/mvtec_ad/bottle/
     train/good/*.png
     test/good/*.png
     test/broken_large/*.png
     test/broken_small/*.png
     test/contamination/*.png
     ground_truth/broken_large/*_mask.png
     ground_truth/broken_small/*_mask.png
     ground_truth/contamination/*_mask.png
   ```
3. Open `final/01_data_exploration.ipynb` in Colab, mount Drive, and edit `BOTTLE_DIR` to match the actual extracted location.
4. Run the notebook from top to bottom. The expected outcome is a display of a normal sample, a defective sample and its matching ground-truth mask; a summary of file counts; and a `(3, 224, 224)` input tensor check.

The official normal training images will later be used to train the student network and create synthetic modifications. Real defective test images and their ground-truth masks are for evaluation, not neural-network training. We inspect a few test examples here solely to understand the dataset.



Stage 2 is loading the frozen DINOv3 teacher and extracting spatial features. We'll add that code only after Stage 1 is verified.

