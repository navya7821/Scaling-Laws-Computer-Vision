# CV-Scaling-Laws

**[Full Report (PDF)](Report.pdf)**  
Empirical study of scaling laws in computer vision using ResNet-18/50/101 on ImageNet single-class subsets (100/200/300 images of tench fish, n01494745). Froze pretrained backbones and trained only a modified FC layer with MSE loss (identity-style regression) to generate comparable per-image "pre-loss" values, then fitted power-law curves to extract scaling exponents (α).

**Key Findings**  
- Deeper models (ResNet50 & 101) consistently show lower median/max per-image loss than ResNet18.  
- Dataset scaling yields near-zero exponent (α ≈ -0.02), indicating diminishing returns beyond ~200 images.  
- High per-image variance suggests preprocessing and selective augmentation are critical on small datasets.

## Repository Contents
- `Scaling_Laws_Report.pdf` – Complete 13-page report  
- `pipeline.py` – Object detection & preprocessing with SAM (ViT-B), Otsu’s thresholding, morphological operations, HSV filtering  
- `model_size_vs_loss.py` – Trains ResNet-18/50/101 on 200-image set  
- `dataset_size_vs_loss.py` – Trains ResNet18 on 100/200/300-image sets  
- `model_size_loss.csv` – Columns: `image_name`, `loss_18`, `loss_50`, `loss_101`  
- `dataset_size_loss.csv` – Columns: `image_name`, `loss_100`, `loss_200`, `loss_300`


