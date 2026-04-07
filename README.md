# Representation Scaling in Computer Vision

This repository contains my internship work focused on understanding how internal representations scale in convolutional vision models under controlled conditions.

The project evolved across two phases, moving from loss-based analysis to deeper representation-level investigation.

## Project Overview

Traditional scaling studies often rely on task-level metrics such as accuracy or loss.  
This work instead explores **representation-level scaling**, analyzing how internal feature behavior changes with:

- Model depth  
- Dataset size  
- Architectural design  

The experiments are conducted under controlled settings using frozen backbones to isolate scaling effects.


## Phases

### Phase 1 — Loss-Based Scaling Analysis
- ResNet-18, ResNet-50, ResNet-101  
- Dataset sizes: 100, 200, 300  
- Metric: per-image loss values  
- Focus: empirical scaling behavior using power-law fitting  

📁 See: `phase1/`


### Phase 2 — Feature Norm Scaling Analysis
- ResNet and DenseNet families  
- Dataset and model scaling  
- Metric: ℓ2 norm of feature representations  
- Focus: representation stability, variance, and scaling behavior  

📁 See: `phase2/`


## Key Insights

- Representation scaling is **strongly architecture-dependent**  
- Model depth influences feature behavior more than dataset size  
- DenseNet exhibits **stable, depth-invariant representations**  
- ResNet shows **high variance and depth-sensitive scaling behavior**


## Research Output

This work was further extended into a structured research study:

- Includes additional architectures (e.g., Vision Transformers)  
- Incorporates multiple representation metrics (ℓ2 norm, variance, effective rank)  
- Provides a more comprehensive analysis of scaling behavior  

📄 **Publication Repository:** https://github.com/navya7821/representation-scaling-vision-models


## Repository Structure
- phase1/ → Initial loss-based scaling experiments
- phase2/ → Feature norm scaling and extended analysis



## Notes

This repository represents the **development and progression** of the work.  
The research version (paper + reproducible code) is maintained separately.
