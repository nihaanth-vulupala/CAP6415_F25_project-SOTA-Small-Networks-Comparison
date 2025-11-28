# Week 4 Training Experiments

## Overview
Training all 4 models on all 3 datasets (12 combinations total)

## Models
1. MobileNetV3-Small (1.6M params, 6.18 MB)
2. EfficientNet-B0 (4.1M params, 15.78 MB)
3. ShuffleNetV2 (444K params, 1.69 MB)
4. SqueezeNet (774K params, 2.95 MB)

## Datasets
1. CIFAR-100: 100 classes, 45000 train / 5000 val / 10000 test
2. Stanford Dogs: 120 classes, 14405 train / 3087 val / 3088 test
3. Flowers-102: 102 classes, 1020 train / 1020 val / 6149 test

## Training Configuration
- Epochs: 15 (with early stopping, patience=5)
- Batch size: 128
- Initial learning rate: 0.01
- Optimizer: SGD (momentum 0.9, weight decay 5e-4)
- LR Scheduler: CosineAnnealingLR
- Pretrained: ImageNet weights
- Device: MPS (Apple Silicon GPU)

## Running Experiments

### Option 1: Run all experiments automatically
```bash
python run_experiments.py
```

### Option 2: Run all experiments with shell script
```bash
./run_all_experiments.sh
```

### Option 3: Run individual experiments

#### MobileNetV3
```bash
python train.py --model mobilenetv3 --dataset cifar100 --epochs 15 --patience 5 --pretrained
python train.py --model mobilenetv3 --dataset stanford_dogs --epochs 15 --patience 5 --pretrained
python train.py --model mobilenetv3 --dataset flowers102 --epochs 15 --patience 5 --pretrained
```

#### EfficientNet-B0
```bash
python train.py --model efficientnet --dataset cifar100 --epochs 15 --patience 5 --pretrained
python train.py --model efficientnet --dataset stanford_dogs --epochs 15 --patience 5 --pretrained
python train.py --model efficientnet --dataset flowers102 --epochs 15 --patience 5 --pretrained
```

#### ShuffleNetV2
```bash
python train.py --model shufflenet --dataset cifar100 --epochs 15 --patience 5 --pretrained
python train.py --model shufflenet --dataset stanford_dogs --epochs 15 --patience 5 --pretrained
python train.py --model shufflenet --dataset flowers102 --epochs 15 --patience 5 --pretrained
```

#### SqueezeNet
```bash
python train.py --model squeezenet --dataset cifar100 --epochs 15 --patience 5 --pretrained
python train.py --model squeezenet --dataset stanford_dogs --epochs 15 --patience 5 --pretrained
python train.py --model squeezenet --dataset flowers102 --epochs 15 --patience 5 --pretrained
```

## Output
- Checkpoints saved to: `checkpoints/{dataset}/{model}/best_model.pth`
- Training logs: Console output with progress bars
- Results: Can redirect to `results/` directory with `tee`

## Estimated Time
Based on test run (3 epochs in 8 minutes):
- Each experiment: 30-40 minutes for 15 epochs (may stop early with early stopping)
- Total for all 12: 6-8 hours if run sequentially
- Recommendation: Run in background or while working on other tasks

## Checkpoints
Best model for each combination saved at:
```
checkpoints/cifar100/mobilenetv3/best_model.pth
checkpoints/cifar100/efficientnet/best_model.pth
checkpoints/cifar100/shufflenet/best_model.pth
checkpoints/cifar100/squeezenet/best_model.pth
checkpoints/stanford_dogs/mobilenetv3/best_model.pth
checkpoints/stanford_dogs/efficientnet/best_model.pth
checkpoints/stanford_dogs/shufflenet/best_model.pth
checkpoints/stanford_dogs/squeezenet/best_model.pth
checkpoints/flowers102/mobilenetv3/best_model.pth
checkpoints/flowers102/efficientnet/best_model.pth
checkpoints/flowers102/shufflenet/best_model.pth
checkpoints/flowers102/squeezenet/best_model.pth
```

## Next Steps After Training
1. Evaluate all models on test sets
2. Compare accuracy, speed, and model size
3. Generate comparison charts
4. Document results in Week 4 log
