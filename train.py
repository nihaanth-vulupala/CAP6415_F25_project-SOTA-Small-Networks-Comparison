"""
Training script for SOTA Small Networks Comparison

This script implements the complete training pipeline for comparing small neural network
architectures (MobileNetV3, EfficientNet-B0, ShuffleNetV2, SqueezeNet) on various image
classification datasets.

Key Features:
    - Transfer learning with ImageNet pretrained weights
    - Cosine annealing learning rate schedule for smooth convergence
    - Early stopping to prevent overfitting
    - Automatic checkpoint saving of best models
    - Support for CUDA, MPS (Apple Silicon), and CPU devices
    - Progress tracking with tqdm progress bars

Training Strategy:
    - Uses SGD optimizer with momentum (0.9) and weight decay (5e-4)
    - Cosine annealing reduces learning rate smoothly over epochs
    - Early stopping monitors validation loss with configurable patience
    - Best model selected based on validation accuracy

Usage:
    python train.py --model mobilenetv3 --dataset cifar100 --epochs 15 --pretrained
    python train.py --model efficientnet --dataset stanford_dogs --epochs 15 --pretrained
    python train.py --model shufflenet --dataset flowers102 --epochs 15 --pretrained
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models import get_model, print_model_summary
from utils.data_loader import get_dataloaders, load_config


class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting

    Monitors validation loss and stops training when it stops improving for a specified
    number of epochs (patience). This prevents the model from overfitting to the training
    data and saves computational resources.

    Args:
        patience (int): Number of epochs to wait for improvement before stopping.
                       Higher values allow more time for recovery from plateaus.
                       Default: 15
        min_delta (float): Minimum change in validation loss to qualify as improvement.
                          Helps ignore trivial fluctuations. Default: 0.001

    Attributes:
        counter (int): Counts epochs without improvement
        best_loss (float): Best validation loss observed so far
        early_stop (bool): Flag indicating whether to stop training
    """

    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience  # How many epochs to wait before stopping
        self.min_delta = min_delta  # Minimum improvement threshold
        self.counter = 0  # Tracks epochs without improvement
        self.best_loss = None  # Stores best validation loss
        self.early_stop = False  # Flag to signal stopping

    def __call__(self, val_loss):
        """
        Check if training should stop based on validation loss

        Args:
            val_loss (float): Current validation loss
        """
        # First epoch - initialize best loss
        if self.best_loss is None:
            self.best_loss = val_loss
        # Loss did not improve by at least min_delta
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            # Patience exhausted - trigger early stopping
            if self.counter >= self.patience:
                self.early_stop = True
        # Loss improved - reset counter and update best loss
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train the model for one epoch

    Performs a full pass through the training dataset, computing gradients and updating
    model weights using backpropagation. Tracks loss and accuracy metrics throughout.

    Training Process:
        1. Set model to training mode (enables dropout, batch norm training)
        2. For each batch:
           - Move data to device (GPU/MPS/CPU)
           - Forward pass: compute predictions
           - Compute loss using cross-entropy
           - Backward pass: compute gradients
           - Update weights using optimizer
        3. Track running metrics and display progress

    Args:
        model (nn.Module): Neural network model to train
        train_loader (DataLoader): DataLoader for training data
        criterion (nn.Module): Loss function (CrossEntropyLoss)
        optimizer (Optimizer): Optimization algorithm (SGD or Adam)
        device (torch.device): Device to run training on (cuda/mps/cpu)
        epoch (int): Current epoch number (for display)

    Returns:
        tuple: (epoch_loss, epoch_acc)
            - epoch_loss (float): Average loss across all batches
            - epoch_acc (float): Training accuracy as percentage (0-100)
    """
    model.train()  # Enable training mode (dropout, batch norm updates)

    running_loss = 0.0  # Accumulator for batch losses
    correct = 0  # Count of correct predictions
    total = 0  # Total number of samples processed

    # Progress bar for visual feedback
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')

    for batch_idx, (images, labels) in enumerate(pbar):
        # Move batch to device (GPU/MPS/CPU)
        images, labels = images.to(device), labels.to(device)

        # Forward pass: compute model predictions
        optimizer.zero_grad()  # Clear gradients from previous batch
        outputs = model(images)  # Get model predictions
        loss = criterion(outputs, labels)  # Compute cross-entropy loss

        # Backward pass: compute gradients and update weights
        loss.backward()  # Backpropagate to compute gradients
        optimizer.step()  # Update model parameters

        # Calculate metrics for this batch
        running_loss += loss.item()  # Accumulate loss
        _, predicted = outputs.max(1)  # Get predicted class (highest logit)
        total += labels.size(0)  # Count samples
        correct += predicted.eq(labels).sum().item()  # Count correct predictions

        # Update progress bar with running metrics
        avg_loss = running_loss / (batch_idx + 1)
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'loss': f'{avg_loss:.3f}',
            'acc': f'{accuracy:.2f}%'
        })

    # Compute final epoch metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch):
    """
    Validate the model on validation set

    Evaluates model performance on validation data without updating weights.
    Used to monitor generalization and detect overfitting during training.

    Validation Process:
        1. Set model to evaluation mode (disables dropout, uses batch norm stats)
        2. Disable gradient computation (saves memory, faster)
        3. For each batch:
           - Move data to device
           - Forward pass only (no backward pass)
           - Compute loss and predictions
        4. Calculate and return metrics

    Args:
        model (nn.Module): Neural network model to validate
        val_loader (DataLoader): DataLoader for validation data
        criterion (nn.Module): Loss function (CrossEntropyLoss)
        device (torch.device): Device to run validation on
        epoch (int): Current epoch number (for display)

    Returns:
        tuple: (epoch_loss, epoch_acc)
            - epoch_loss (float): Average validation loss
            - epoch_acc (float): Validation accuracy as percentage (0-100)
    """
    model.eval()  # Set to evaluation mode (disables dropout, batch norm training)

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]  ')

    # Disable gradient computation for validation (saves memory and computation)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move data to device
            images, labels = images.to(device), labels.to(device)

            # Forward pass only (no gradient computation)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Calculate metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)  # Get predicted class
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar with running metrics
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.3f}',
                'acc': f'{accuracy:.2f}%'
            })

    # Compute final validation metrics
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path):
    """
    Save model checkpoint to disk

    Saves complete training state including model weights, optimizer state, and metrics.
    This allows resuming training or loading the best model for evaluation.

    Checkpoint Contents:
        - epoch: Training epoch when checkpoint was saved
        - model_state_dict: Model weights and parameters
        - optimizer_state_dict: Optimizer state (momentum buffers, etc.)
        - val_acc: Validation accuracy at this checkpoint

    Args:
        model (nn.Module): Trained model to save
        optimizer (Optimizer): Optimizer with current state
        epoch (int): Current epoch number
        val_acc (float): Validation accuracy percentage
        checkpoint_path (str): Path where checkpoint will be saved

    Note:
        Creates parent directories if they don't exist
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Package all training state into checkpoint dictionary
    checkpoint = {
        'epoch': epoch,  # Which epoch this checkpoint is from
        'model_state_dict': model.state_dict(),  # Model weights
        'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
        'val_acc': val_acc  # Validation accuracy for this checkpoint
    }

    # Save checkpoint to disk
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved: {checkpoint_path}')


def train(args):
    """
    Main training function - orchestrates the complete training pipeline

    This function coordinates all training components: data loading, model creation,
    optimizer setup, training loop, validation, checkpointing, and early stopping.

    Training Pipeline:
        1. Setup: Load config, detect device, prepare data
        2. Model: Create model with pretrained weights (if specified)
        3. Optimization: Configure SGD/Adam with cosine annealing LR schedule
        4. Training Loop:
           - Train one epoch
           - Validate on validation set
           - Save checkpoint if best model
           - Check early stopping criteria
        5. Complete: Save final statistics

    Args:
        args (Namespace): Command-line arguments containing:
            - model: Architecture name (mobilenetv3, efficientnet, etc.)
            - dataset: Dataset name (cifar100, stanford_dogs, flowers102)
            - epochs: Maximum number of training epochs
            - lr: Initial learning rate
            - optimizer: Optimization algorithm (sgd or adam)
            - patience: Early stopping patience
            - pretrained: Whether to use ImageNet pretrained weights
    """

    # ========== SETUP PHASE ==========
    # Load configuration file with dataset and training parameters
    config = load_config()

    # Detect and setup compute device (CUDA GPU > MPS (Apple Silicon) > CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')  # Use Apple Silicon GPU if available
    print(f'Using device: {device}')

    # Get number of output classes for this dataset
    num_classes = config['datasets'][args.dataset]['num_classes']

    # ========== DATA LOADING ==========
    # Load train/val/test dataloaders with augmentation
    print(f'\nLoading {args.dataset} dataset...')
    train_loader, val_loader, test_loader = get_dataloaders(args.dataset, config)

    # ========== MODEL CREATION ==========
    # Create model with ImageNet pretrained weights (transfer learning)
    print(f'\nCreating {args.model} model...')
    model = get_model(args.model, num_classes=num_classes, pretrained=args.pretrained)
    model = model.to(device)  # Move model to GPU/MPS/CPU
    print_model_summary(model, args.model)  # Display architecture info

    # ========== OPTIMIZATION SETUP ==========
    # Loss function: Cross-entropy for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Optimizer: SGD with momentum or Adam
    if args.optimizer == 'sgd':
        # SGD: Stochastic Gradient Descent with momentum
        # Momentum helps accelerate convergence and dampen oscillations
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,  # Initial learning rate
            momentum=config['training']['momentum'],  # Momentum factor (0.9)
            weight_decay=config['training']['weight_decay']  # L2 regularization (5e-4)
        )
    else:
        # Adam: Adaptive moment estimation (alternative optimizer)
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=config['training']['weight_decay']
        )

    # Learning rate scheduler: Cosine annealing for smooth LR decay
    # Gradually reduces LR from initial value to near-zero following cosine curve
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Early stopping: Prevents overfitting by stopping when validation stops improving
    early_stopping = EarlyStopping(patience=args.patience)

    # ========== CHECKPOINT SETUP ==========
    # Prepare directory for saving model checkpoints
    checkpoint_dir = os.path.join('checkpoints', args.dataset, args.model)
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')

    # ========== TRAINING LOOP ==========
    print(f'\nStarting training for {args.epochs} epochs...\n')

    best_val_acc = 0.0  # Track best validation accuracy
    training_start = time.time()  # Record training start time

    # Loop through epochs (1 to max_epochs or until early stopping)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()  # Time this epoch

        # TRAIN: One full pass through training data
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # VALIDATE: Evaluate on validation set (no gradient updates)
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )

        # UPDATE LEARNING RATE: Step the cosine annealing scheduler
        scheduler.step()  # Decay learning rate
        current_lr = scheduler.get_last_lr()[0]  # Get current LR for logging

        # EPOCH SUMMARY: Print statistics for this epoch
        epoch_time = time.time() - epoch_start
        print(f'\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s):')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%')
        print(f'  LR: {current_lr:.6f}')

        # CHECKPOINT: Save model if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, best_checkpoint_path)
            print(f'  New best validation accuracy: {best_val_acc:.2f}%')

        # EARLY STOPPING: Check if we should stop training
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f'\nEarly stopping triggered at epoch {epoch}')
            print(f'Validation loss has not improved for {args.patience} epochs')
            break

        print('-' * 70)

    training_time = time.time() - training_start

    # final summary
    print('\n' + '=' * 70)
    print('TRAINING COMPLETE')
    print('=' * 70)
    print(f'Total training time: {training_time/60:.1f} minutes')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    print(f'Best model saved to: {best_checkpoint_path}')
    print('=' * 70 + '\n')


def main():
    parser = argparse.ArgumentParser(description='Train SOTA small networks')

    # model and dataset
    parser.add_argument('--model', type=str, required=True,
                        choices=['mobilenetv3', 'efficientnet', 'shufflenet', 'squeezenet'],
                        help='Model to train')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar100', 'stanford_dogs', 'flowers102'],
                        help='Dataset to use')

    # training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'],
                        help='Optimizer (default: sgd)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='Train from scratch')
    parser.set_defaults(pretrained=True)

    args = parser.parse_args()

    print('\n' + '=' * 70)
    print('TRAINING CONFIGURATION')
    print('=' * 70)
    print(f'Model: {args.model}')
    print(f'Dataset: {args.dataset}')
    print(f'Epochs: {args.epochs}')
    print(f'Learning rate: {args.lr}')
    print(f'Optimizer: {args.optimizer}')
    print(f'Pretrained: {args.pretrained}')
    print(f'Early stopping patience: {args.patience}')
    print('=' * 70 + '\n')

    train(args)


if __name__ == '__main__':
    main()
