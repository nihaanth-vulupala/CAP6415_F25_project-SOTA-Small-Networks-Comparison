"""
Evaluation Script for SOTA Small Networks Comparison

This script evaluates all trained model checkpoints on their respective test sets,
measuring accuracy, inference time, model size, and other performance metrics.

Key Features:
    - Test accuracy measurement on held-out test data
    - Inference time benchmarking with device synchronization
    - Model size and parameter count analysis
    - Automated evaluation of all 12 model-dataset combinations
    - JSON output for easy comparison and visualization

Evaluation Metrics:
    - Test Accuracy: Classification accuracy on test set (%)
    - Validation Accuracy: Best validation accuracy during training (%)
    - Inference Time: Average time per batch in milliseconds
    - Model Size: Memory footprint in megabytes
    - Total Parameters: Number of trainable parameters

Usage:
    # Evaluate all trained models
    python evaluate.py

    # Evaluate specific model and dataset
    python evaluate.py --model mobilenetv3 --dataset cifar100

Output:
    - results/evaluation_results.json: Complete results for all models
    - Console output: Per-model evaluation statistics
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
    """
    Evaluate model classification accuracy on test set

    Performs inference on the entire test set and computes top-1 accuracy.
    Uses gradient-free evaluation mode for efficiency.

    Process:
        1. Set model to evaluation mode (disables dropout, batch norm training)
        2. Iterate through test set batches
        3. For each batch:
           - Forward pass (no gradients)
           - Get predicted class (argmax of logits)
           - Count correct predictions
        4. Compute overall accuracy percentage

    Args:
        model (nn.Module): Trained neural network model
        test_loader (DataLoader): DataLoader for test dataset
        device (torch.device): Device to run evaluation on (cuda/mps/cpu)

    Returns:
        float: Test accuracy as percentage (0-100)
    """
    model.eval()  # Set to evaluation mode
    correct = 0  # Count of correct predictions
    total = 0  # Total samples evaluated

    # Disable gradient computation for efficiency
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            # Move batch to device
            images, labels = images.to(device), labels.to(device)

            # Forward pass: get model predictions
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # Get class with highest score

            # Update counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy percentage
    accuracy = 100 * correct / total
    return accuracy

def measure_inference_time(model, test_loader, device, num_batches=50):
    """
    Measure average inference time per batch

    Benchmarks model inference speed by measuring the time taken for forward passes
    on multiple batches. Uses device synchronization to ensure accurate timing.

    Why Synchronization?
        GPU/MPS operations are asynchronous - they return before completing. Without
        synchronization, we'd measure scheduling time, not actual computation time.
        Synchronization ensures all operations finish before timing stops.

    Process:
        1. Set model to evaluation mode
        2. For each batch (up to num_batches):
           - Move data to device
           - Record start time
           - Forward pass
           - Synchronize device (wait for completion)
           - Record end time
        3. Calculate average across all batches

    Args:
        model (nn.Module): Model to benchmark
        test_loader (DataLoader): DataLoader providing test batches
        device (torch.device): Compute device (cuda/mps/cpu)
        num_batches (int): Number of batches to average over (default: 50)

    Returns:
        float: Average inference time per batch in milliseconds
    """
    model.eval()  # Set to evaluation mode
    times = []  # Store time for each batch

    # Disable gradients for inference
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            # Only benchmark specified number of batches
            if i >= num_batches:
                break

            # Move batch to device
            images = images.to(device)

            # Time the forward pass with device synchronization
            start = time.time()
            _ = model(images)  # Forward pass
            # Synchronize to ensure computation completes before timing
            if device.type == 'mps':
                torch.mps.synchronize()  # Apple Silicon GPU sync
            elif device.type == 'cuda':
                torch.cuda.synchronize()  # NVIDIA GPU sync
            end = time.time()

            # Record batch time
            times.append(end - start)

    # Calculate average time in milliseconds
    avg_time = sum(times) / len(times) * 1000  # Convert seconds to ms
    return avg_time

def evaluate_checkpoint(model_name, dataset_name, config, device):
    """
    Evaluate a single trained model checkpoint comprehensively

    Loads a trained model from checkpoint and performs complete evaluation including
    test accuracy, inference time, and model statistics. This function coordinates
    all evaluation metrics for a single model-dataset combination.

    Evaluation Steps:
        1. Locate and load checkpoint file
        2. Create model architecture matching checkpoint
        3. Load trained weights from checkpoint
        4. Load test dataset
        5. Measure test accuracy (classification performance)
        6. Measure inference time (speed benchmark)
        7. Calculate model size and parameter count
        8. Package results into dictionary

    Args:
        model_name (str): Model architecture (mobilenetv3, efficientnet, etc.)
        dataset_name (str): Dataset name (cifar100, stanford_dogs, flowers102)
        config (dict): Configuration dictionary with dataset/training parameters
        device (torch.device): Device for evaluation (cuda/mps/cpu)

    Returns:
        dict or None: Evaluation results containing:
            - model: Model architecture name
            - dataset: Dataset name
            - test_accuracy: Test set accuracy (%)
            - val_accuracy: Best validation accuracy during training (%)
            - total_params: Total number of parameters
            - model_size_mb: Model memory footprint (MB)
            - avg_inference_time_ms: Average inference time (milliseconds)
            - checkpoint_epoch: Training epoch when checkpoint was saved
        Returns None if checkpoint file not found.
    """
    # Construct checkpoint path
    checkpoint_path = f'checkpoints/{dataset_name}/{model_name}/best_model.pth'

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    print(f"\nEvaluating {model_name} on {dataset_name}")
    print(f"Checkpoint: {checkpoint_path}")

    # ========== MODEL SETUP ==========
    # Map dataset names to number of classes
    dataset_info = {
        'cifar100': 100,  # 100 object categories
        'stanford_dogs': 120,  # 120 dog breeds
        'flowers102': 102  # 102 flower species
    }
    num_classes = dataset_info[dataset_name]

    # Create model architecture (no pretrained weights needed)
    model = get_model(model_name, num_classes=num_classes, pretrained=False)

    # Load checkpoint and restore trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])  # Load trained parameters
    model = model.to(device)  # Move to evaluation device

    # ========== DATA LOADING ==========
    # Get test dataloader (we only need test set for evaluation)
    _, _, test_loader = get_dataloaders(dataset_name, config)

    # ========== MODEL ANALYSIS ==========
    # Count parameters and calculate model size
    param_info = count_parameters(model)
    total_params = param_info['total']  # Total trainable parameters
    model_size_mb = param_info['total_mb']  # Memory footprint in MB

    # ========== ACCURACY EVALUATION ==========
    # Measure classification accuracy on test set
    print("Measuring test accuracy...")
    test_accuracy = evaluate_model(model, test_loader, device)

    # ========== SPEED BENCHMARK ==========
    # Measure average inference time per batch
    print("Measuring inference time...")
    avg_inference_time = measure_inference_time(model, test_loader, device)

    # Retrieve validation accuracy from checkpoint (best accuracy during training)
    val_accuracy = checkpoint.get('val_accuracy', 0.0)

    # ========== PACKAGE RESULTS ==========
    # Compile all metrics into results dictionary
    results = {
        'model': model_name,  # Architecture name
        'dataset': dataset_name,  # Dataset name
        'test_accuracy': round(test_accuracy, 2),  # Test accuracy percentage
        'val_accuracy': round(val_accuracy, 2),  # Best validation accuracy
        'total_params': total_params,  # Total parameter count
        'model_size_mb': round(model_size_mb, 2),  # Model size in MB
        'avg_inference_time_ms': round(avg_inference_time, 2),  # Speed in ms/batch
        'checkpoint_epoch': checkpoint.get('epoch', 0)  # Training epoch
    }

    # Display summary to console
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
