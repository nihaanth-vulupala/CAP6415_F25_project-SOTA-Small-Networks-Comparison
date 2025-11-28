"""
Evaluation script for trained models.
Tests all trained model checkpoints and generates comparison results.
"""

import torch
import argparse
import os
import time
import json
from tqdm import tqdm
from models import get_model, count_parameters
from utils.data_loader import get_dataloaders, load_config

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def measure_inference_time(model, test_loader, device, num_batches=50):
    """Measure average inference time."""
    model.eval()
    times = []

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_batches:
                break

            images = images.to(device)

            start = time.time()
            _ = model(images)
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()

            times.append(end - start)

    avg_time = sum(times) / len(times) * 1000
    return avg_time

def evaluate_checkpoint(model_name, dataset_name, config, device):
    """Evaluate a single checkpoint."""
    checkpoint_path = f'checkpoints/{dataset_name}/{model_name}/best_model.pth'

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    print(f"\nEvaluating {model_name} on {dataset_name}")
    print(f"Checkpoint: {checkpoint_path}")

    # Get dataset info
    dataset_info = {
        'cifar100': 100,
        'stanford_dogs': 120,
        'flowers102': 102
    }
    num_classes = dataset_info[dataset_name]

    # Load model
    model = get_model(model_name, num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Get data loaders
    _, _, test_loader = get_dataloaders(dataset_name, config)

    # Get model info
    param_info = count_parameters(model)
    total_params = param_info['total']
    model_size_mb = param_info['total_mb']

    # Evaluate accuracy
    print("Measuring test accuracy...")
    test_accuracy = evaluate_model(model, test_loader, device)

    # Measure inference time
    print("Measuring inference time...")
    avg_inference_time = measure_inference_time(model, test_loader, device)

    val_accuracy = checkpoint.get('val_accuracy', 0.0)

    results = {
        'model': model_name,
        'dataset': dataset_name,
        'test_accuracy': round(test_accuracy, 2),
        'val_accuracy': round(val_accuracy, 2),
        'total_params': total_params,
        'model_size_mb': round(model_size_mb, 2),
        'avg_inference_time_ms': round(avg_inference_time, 2),
        'checkpoint_epoch': checkpoint.get('epoch', 0)
    }

    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Val Accuracy: {val_accuracy:.2f}%")
    print(f"Parameters: {total_params:,}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"Inference Time: {avg_inference_time:.2f} ms/batch")

    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model', type=str, choices=['mobilenetv3', 'efficientnet', 'shufflenet', 'squeezenet', 'all'],
                        default='all', help='Model to evaluate')
    parser.add_argument('--dataset', type=str, choices=['cifar100', 'stanford_dogs', 'flowers102', 'all'],
                        default='all', help='Dataset to evaluate on')
    parser.add_argument('--output', type=str, default='results/evaluation_results.json',
                        help='Output file for results')

    args = parser.parse_args()

    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Load config
    config = load_config()

    # Define models and datasets to evaluate
    models = ['mobilenetv3', 'efficientnet', 'shufflenet', 'squeezenet'] if args.model == 'all' else [args.model]
    datasets = ['cifar100', 'stanford_dogs', 'flowers102'] if args.dataset == 'all' else [args.dataset]

    all_results = []

    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)

    for model_name in models:
        for dataset_name in datasets:
            result = evaluate_checkpoint(model_name, dataset_name, config, device)
            if result:
                all_results.append(result)
            print("-"*80)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    # Print summary table
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Model':<15} {'Dataset':<15} {'Test Acc':<10} {'Val Acc':<10} {'Params':<12} {'Size (MB)':<10} {'Time (ms)':<10}")
    print("-"*80)

    for result in all_results:
        print(f"{result['model']:<15} {result['dataset']:<15} {result['test_accuracy']:<10.2f} "
              f"{result['val_accuracy']:<10.2f} {result['total_params']:<12,} "
              f"{result['model_size_mb']:<10.2f} {result['avg_inference_time_ms']:<10.2f}")

    print("="*80)

if __name__ == '__main__':
    main()
