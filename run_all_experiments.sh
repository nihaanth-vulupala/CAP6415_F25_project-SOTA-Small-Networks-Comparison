#!/bin/bash
# Week 4 Training Experiments
# Run all 12 model-dataset combinations

echo "Starting Week 4 Training Experiments"
echo "Total: 12 model-dataset combinations"
echo "Each will train for up to 15 epochs with early stopping"
echo ""

# Create results directory
mkdir -p results

# MobileNetV3 experiments
echo "=== MobileNetV3 Experiments ==="
.venv/bin/python train.py --model mobilenetv3 --dataset cifar100 --epochs 15 --patience 5 --pretrained 2>&1 | tee results/mobilenetv3_cifar100.log
.venv/bin/python train.py --model mobilenetv3 --dataset stanford_dogs --epochs 15 --patience 5 --pretrained 2>&1 | tee results/mobilenetv3_stanford_dogs.log
.venv/bin/python train.py --model mobilenetv3 --dataset flowers102 --epochs 15 --patience 5 --pretrained 2>&1 | tee results/mobilenetv3_flowers102.log

# EfficientNet-B0 experiments
echo "=== EfficientNet-B0 Experiments ==="
.venv/bin/python train.py --model efficientnet --dataset cifar100 --epochs 15 --patience 5 --pretrained 2>&1 | tee results/efficientnet_cifar100.log
.venv/bin/python train.py --model efficientnet --dataset stanford_dogs --epochs 15 --patience 5 --pretrained 2>&1 | tee results/efficientnet_stanford_dogs.log
.venv/bin/python train.py --model efficientnet --dataset flowers102 --epochs 15 --patience 5 --pretrained 2>&1 | tee results/efficientnet_flowers102.log

# ShuffleNetV2 experiments
echo "=== ShuffleNetV2 Experiments ==="
.venv/bin/python train.py --model shufflenet --dataset cifar100 --epochs 15 --patience 5 --pretrained 2>&1 | tee results/shufflenet_cifar100.log
.venv/bin/python train.py --model shufflenet --dataset stanford_dogs --epochs 15 --patience 5 --pretrained 2>&1 | tee results/shufflenet_stanford_dogs.log
.venv/bin/python train.py --model shufflenet --dataset flowers102 --epochs 15 --patience 5 --pretrained 2>&1 | tee results/shufflenet_flowers102.log

# SqueezeNet experiments
echo "=== SqueezeNet Experiments ==="
.venv/bin/python train.py --model squeezenet --dataset cifar100 --epochs 15 --patience 5 --pretrained 2>&1 | tee results/squeezenet_cifar100.log
.venv/bin/python train.py --model squeezenet --dataset stanford_dogs --epochs 15 --patience 5 --pretrained 2>&1 | tee results/squeezenet_stanford_dogs.log
.venv/bin/python train.py --model squeezenet --dataset flowers102 --epochs 15 --patience 5 --pretrained 2>&1 | tee results/squeezenet_flowers102.log

echo ""
echo "All experiments completed!"
echo "Logs saved in results/ directory"
