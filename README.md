# CAP6415_F25_project-SOTA-Small-Networks-Comparison

## Abstract

**Problem:** There is a need to understand the trade-offs among accuracy, model size, and inference speed when determining which neural network architecture to use in order to optimize the performance of resource-constrained edge devices. Although small networks that are currently considered to be state-of-the-art in terms of efficiency were developed primarily with this objective in mind, it has been unclear how well they perform in relation to one another on a variety of different datasets as well as in various deployment settings.

**Solution:** The goal of this research project was to conduct a broad comparative study of the performance characteristics of four commonly used CNN architectures (SqueezeNet, ShuffleNet V2, MobileNet v3 and EfficientNet B0) on three challenging image classification datasets (CIFAR-100, Stanford Dogs, Flowers-102). We conducted our evaluation using the same transfer learning approach that utilized pre-trained ImageNet weights, comparing each of the models on the basis of their test accuracy, inference times, model sizes and number of parameters. The overall findings of this study demonstrate that for the majority of deployment scenarios, MobileNet V3 represents the best balance between accuracy and efficiency (79.26% test accuracy on CIFAR-100, 6.18 MB model size); however, the results also show that while Fine-Grained tasks can achieve higher levels of accuracy with larger models (e.g., 89.40% on Flowers-102 with an average model size of 15.78 MB), the larger models require additional resources that may not be available in all types of deployments. Overall, these results provide researchers and practitioners with relevant information about the relative strengths and weaknesses of these models for mobile, edge, and cloud-based deployment applications.

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

## For TA: Reproducibility Instructions

### Quick Setup (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/nihaanth-vulupala/CAP6415_F25_project-SOTA-Small-Networks-Comparison.git
cd CAP6415_F25_project-SOTA-Small-Networks-Comparison

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run Evaluation (Recommended - 5-10 minutes)

Evaluate all 12 trained models using existing checkpoints:

```bash
# Evaluate all models (datasets auto-download on first run)
python evaluate.py

# This will:
# - Load all 12 trained model checkpoints
# - Download datasets automatically (CIFAR-100, Stanford Dogs, Flowers-102) - ~1.2 GB
# - Evaluate test accuracy and inference time
# - Generate results/evaluation_results.json
```

**Expected Output:**
- `results/evaluation_results.json` - Contains all 12 model results
- Console output showing accuracy for each model-dataset combination

**Expected Results:**
- MobileNetV3 + CIFAR-100: **79.26%** accuracy
- EfficientNet + Flowers-102: **89.40%** accuracy
- EfficientNet + Stanford Dogs: **72.12%** accuracy

**Runtime:** ~5-10 minutes (first run includes dataset download)

### Generate Visualizations

Create all comparison charts and analysis plots:

```bash
python create_demo_visualizations.py

# Generates ten visualization in the folder results/figures/:
# - accuracy_comparison.png
# - architecture_comparison.png
# - efficiency_comparison.png
# - inference_time_comparison.png
# - performance_heatmap.png
# - efficiency_frontier.png
# - deployment_recommendations.png
# - dataset_analysis.png
# - comparison_table.png
# - summary_dashboard.png
```

**Runtime:** ~30 seconds

### Test Single Model (Optional - 30-60 minutes)

To verify the training pipeline works, train one model:

```bash
# Train MobileNetV3 on CIFAR-100 (fastest experiment)
python train.py --model mobilenetv3 --dataset cifar100 --epochs 15 --pretrained

# Expected runtime:
# - GPU (CUDA/MPS): ~30 minutes
# - CPU: ~2 hours
# Expected accuracy: ~79% (may vary ±2% due to randomness)
```

**Other model options:**
```bash
# Train EfficientNet on Flowers-102
python train.py --model efficientnet --dataset flowers102 --epochs 15 --pretrained

# Train ShuffleNet on Stanford Dogs
python train.py --model shufflenet --dataset stanford_dogs --epochs 15 --pretrained

# Train SqueezeNet on CIFAR-100
python train.py --model squeezenet --dataset cifar100 --epochs 15 --pretrained
```

### Verification Checklist

After running the evaluation, verify:
- [ ] `results/evaluation_results.json` exists with 12 entries
- [ ] MobileNetV3 achieves ~79% on CIFAR-100
- [ ] EfficientNet achieves ~89% on Flowers-102
- [ ] All 10 PNG visualizations are generated in `results/figures/`

### Trouble shooting
**"ModuleNotFoundError: No module named 'torch'".**

Solution: Make sure to activate the virtual environment with `source venv/bin/activate`.

**"FileNotFoundError: checkpoint not found".**

Solution: Make sure you are in the project directory.

**"CUDA out of memory".** (During Training)

Solution: Add `--device cpu` flag or decrease batch size in `configs/config.yaml`.

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
Z32815779
Computer Vision CAP6415 - Fall 2025
Florida Atlantic University
