# Feature Norm Scaling Analysis in Vision Models

## Overview

This phase of the project investigates how internal feature representations scale in convolutional neural networks under controlled conditions.  
Instead of relying on task-level metrics such as accuracy or loss, the analysis focuses on the **ℓ2 norm of feature representations** to study scaling behavior.

The goal is to understand how **model architecture, depth, and dataset size** influence representation magnitude and stability.


## Experimental Setup

### Architectures
- **ResNet family**: ResNet-18, ResNet-50, ResNet-101  
- **DenseNet family**: DenseNet-121, DenseNet-169, DenseNet-201  

### Scaling Axes
- **Model size scaling**: variation across architecture depth  
- **Dataset size scaling**: 100, 200, 300 samples  

### Metric
- ℓ2 norm of final-layer feature representations  

### Controls
- Frozen backbone networks  
- Deterministic training setup  
- Fixed preprocessing and data ordering  


## Key Findings

### 1. Architecture-dependent scaling behavior
- DenseNet exhibits **stable and consistent feature norms** across depth  
- ResNet shows **high variance and heavy-tailed distributions**, especially in deeper models  

### 2. Stronger sensitivity to model depth than dataset size
- Model scaling produces **significant changes** in representation magnitude  
- Dataset scaling has **minimal impact** within the examined range  

### 3. Scaling law behavior
- ResNet: α ≈ -0.326 → **unstable, depth-sensitive scaling**  
- DenseNet: α ≈ -0.028 → **near-invariant, stable scaling** :contentReference[oaicite:0]{index=0}  

### 4. Representation stability differences
- DenseNet: smooth, bounded, and consistent behavior  
- ResNet: outliers and large variance dominate deeper models  


## Repository Contents

- `feature_norm_scaling_analysis.ipynb` — experiment implementation and analysis  
- `dataset_scaling_*.csv` — dataset size ablation results  
- `model_scaling_*.csv` — model size ablation results  
- `Report.pdf` — detailed analysis and findings  

## Full Report

👉 [View Full Report](./Report.pdf)


## Notes

This phase extends earlier loss-based analysis by focusing on **representation-level scaling**, providing a more controlled and architecture-centric understanding of model behavior.

A more comprehensive study including additional architectures and metrics is available separately in the associated research work.
