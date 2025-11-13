# CAP6415_F25_project-SOTA-Small-Networks-Comparison

## Abstract

This project presents a comprehensive comparison of State-of-the-Art (SOTA) small convolutional neural networks (CNNs) with less than 50 layers. The study evaluates MobileNetV3, EfficientNet-B0, ShuffleNetV2, and SqueezeNet on multiple image classification datasets beyond their original benchmarks. The comparison focuses on key metrics including accuracy, inference time, model size, and computational efficiency (FLOPs). This analysis provides insights into the trade-offs between model complexity and performance for resource-constrained deployment scenarios.

## Problem Statement

While many SOTA models achieve high accuracy on standard benchmarks (ImageNet, CIFAR-10), their performance on diverse datasets and real-world scenarios remains underexplored. This project addresses:
1. How do small SOTA networks perform on datasets different from their original benchmarks?
2. What are the trade-offs between accuracy, speed, and model size across different architectures?
3. Which architecture is most suitable for specific deployment constraints?

## Methodology

### Models Selected
1. **MobileNetV3-Small** (~2.5M parameters, ~15 layers)
2. **EfficientNet-B0** (~5.3M parameters, ~237 layers - using depth-wise separable convolutions)
3. **ShuffleNetV2 0.5x** (~1.4M parameters, ~50 layers)
4. **SqueezeNet 1.1** (~1.2M parameters, ~18 layers)

### Datasets Used
- **CIFAR-100** (60,000 32x32 color images, 100 classes)
- **Stanford Dogs** (20,580 images, 120 dog breeds)
- **Flowers-102** (8,189 images, 102 flower categories)

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
