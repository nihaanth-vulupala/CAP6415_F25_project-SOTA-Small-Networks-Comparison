# CAP6415_F25_project-SOTA-Small-Networks-Comparison

## Abstract

**Problem:** Selecting the optimal neural network architecture for resource-constrained edge devices requires understanding the trade-offs between accuracy, model size, and inference speed. While state-of-the-art small networks are designed for efficiency, their relative performance across diverse datasets and deployment scenarios remains unclear.

**Solution:** This project provides a comprehensive empirical comparison of four efficient CNN architectures (MobileNetV3, EfficientNet-B0, ShuffleNetV2, and SqueezeNet) across three challenging image classification datasets (CIFAR-100, Stanford Dogs, Flowers-102). Using transfer learning with ImageNet-pretrained weights, we evaluate each model on test accuracy, inference time, model size, and parameter count. Our analysis reveals that MobileNetV3 offers the best accuracy-efficiency balance for most scenarios (79.26% on CIFAR-100, 6.18 MB), while EfficientNet achieves the highest accuracy on fine-grained tasks (89.40% on Flowers-102) at the cost of increased size (15.78 MB). The results provide actionable insights for practitioners selecting models for mobile, edge, and cloud deployment contexts.

## Problem Statement

While many SOTA models achieve high accuracy on standard benchmarks (ImageNet, CIFAR-10), their performance on diverse datasets and real-world scenarios remains underexplored. This project addresses:
1. How do small SOTA networks perform on datasets different from their original benchmarks?
2. What are the trade-offs between accuracy, speed, and model size across different architectures?
3. Which architecture is most suitable for specific deployment constraints?

## Methodology

### Models Selected

All selected models qualify as "small networks" based on parameter efficiency and computational cost, designed specifically for resource-constrained deployment:

1. **MobileNetV3-Small** (~2.5M parameters, ~15 layers)
   - Efficient inverted residual architecture
   - Hardware-aware NAS optimization

2. **EfficientNet-B0** (~5.0M parameters, efficient compound-scaled architecture)*
   - Depthwise separable convolutions drastically reduce computational cost
   - Despite higher layer count, maintains low parameter count and FLOPs
   - Baseline of the EfficientNet family, designed for mobile/edge devices

3. **ShuffleNetV2 0.5x** (~1.4M parameters, ~50 layers)
   - Optimized for memory access cost (MAC), not just FLOPs
   - Channel shuffle for efficient information flow

4. **SqueezeNet 1.1** (~1.2M parameters, ~18 layers)
   - Fire modules with aggressive parameter reduction
   - AlexNet-level accuracy with 50x fewer parameters

**Note on "Small Networks":** While EfficientNet-B0 has more conventional layers due to compound scaling, it qualifies as a small network based on: (1) parameter efficiency (5.0M vs. 25M+ for ResNet50), (2) computational efficiency through depthwise separable convolutions, (3) design intent for resource-constrained environments, and (4) mobile/edge deployment suitability. The architecture demonstrates that layer count alone does not determine network size - the type of operations and parameter count are equally important metrics.

### Datasets Used

To ensure rigorous comparison beyond standard leaderboard benchmarks, we evaluate on three diverse datasets with different characteristics:

- **CIFAR-100** (60,000 32x32 color images, 100 classes)
  - General object classification with low resolution
  - Tests model performance on small input sizes

- **Stanford Dogs** (20,580 images, 120 dog breeds)
  - Fine-grained classification task (inter-class similarity)
  - Tests model ability to distinguish subtle visual differences

- **Flowers-102** (8,189 images, 102 flower categories)
  - Fine-grained classification with varying backgrounds
  - Tests generalization with limited training data

These datasets differ significantly from ImageNet (the standard pretraining benchmark), allowing us to evaluate transfer learning effectiveness and model generalization across diverse visual domains. This comparison is NOT a recreation of published leaderboard results.

### Evaluation Metrics
- Top-1 and Top-5 Accuracy
- Inference Time (ms per image)
- Model Size (MB)
- FLOPs (Floating Point Operations)
- Training Time per Epoch
- Memory Usage

## Framework and Dependencies

**Framework:** PyTorch 2.0+

**Key Libraries:**
- torchvision (pre-trained models)
- timm (PyTorch Image Models)
- thop (FLOPs calculation)
- matplotlib, seaborn (visualization)
- pandas, numpy (data processing)

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/CAP6415_F25_project-SOTA-Small-Networks-Comparison.git
cd CAP6415_F25_project-SOTA-Small-Networks-Comparison

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Train all models on CIFAR-100
python train.py --dataset cifar100 --models all

# Train specific model
python train.py --dataset cifar100 --models mobilenetv3

# Evaluate models
python evaluate.py --dataset cifar100 --checkpoint checkpoints/

# Generate comparison plots
python visualize_results.py
```

## Project Structure

```
├── README.md
├── requirements.txt
├── train.py                  # Training script for all models
├── evaluate.py              # Evaluation and benchmarking
├── visualize_results.py     # Generate comparison plots
├── models/
│   ├── __init__.py
│   ├── mobilenetv3.py
│   ├── efficientnet.py
│   ├── shufflenet.py
│   └── squeezenet.py
├── utils/
│   ├── __init__.py
│   ├── data_loader.py       # Dataset loading utilities
│   ├── metrics.py           # Metric calculation
│   └── benchmark.py         # Speed and FLOPs benchmarking
├── configs/
│   └── config.yaml          # Hyperparameters
├── checkpoints/             # Saved model weights
├── results/                 # Experimental results
│   ├── plots/
│   ├── metrics.csv
│   └── analysis.md
├── logs/                    # Training logs
│   ├── week1log.txt
│   ├── week2log.txt
│   ├── week3log.txt
│   ├── week4log.txt
│   └── week5log.txt
└── notebooks/               # Jupyter notebooks for analysis
    └── exploratory_analysis.ipynb
```
## Data Setup
- Placed CIFAR-100 files under `datasets/cifar100/cifar-100-python/`
- Placed Flowers-102 images and label files under `datasets/flowers102/flowers-102/`
- Placed Stanford Dogs images in `datasets/stanford_dogs/Images/{breed_folder}/`
- Do not commit actual data or .venv to GitHub (ensure .gitignore is set)


## Attribution

This project builds upon the following works:

### Models
- **MobileNetV3**: Howard, A., et al. "Searching for MobileNetV3." ICCV 2019.
- **EfficientNet**: Tan, M., & Le, Q. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." ICML 2019.
- **ShuffleNetV2**: Ma, N., et al. "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design." ECCV 2018.
- **SqueezeNet**: Iandola, F., et al. "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters." arXiv 2016.

### Code References
- PyTorch official models: https://github.com/pytorch/vision
- timm library: https://github.com/huggingface/pytorch-image-models
- Dataset implementations from torchvision and PyTorch Hub

## Results

See [results/README.md](results/README.md) for detailed experimental results and analysis.

## License

MIT License - See LICENSE file for details

## Author
Nihaanth Reddy Vulupala.
Computer Vision CAP6415 - Fall 2025
Florida Atlantic University
