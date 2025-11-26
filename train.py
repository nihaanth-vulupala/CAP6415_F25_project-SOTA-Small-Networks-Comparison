"""
Training script for SOTA Small Networks Comparison

Usage:
    python train.py --model mobilenetv3 --dataset cifar100 --epochs 100
    python train.py --model efficientnet --dataset stanford_dogs --epochs 50
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
    """Early stopping to stop training when validation loss doesn't improve"""

    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        loss.backward()
        optimizer.step()

        # metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # update progress bar
        avg_loss = running_loss / (batch_idx + 1)
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'loss': f'{avg_loss:.3f}',
            'acc': f'{accuracy:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]  ')

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.3f}',
                'acc': f'{accuracy:.2f}%'
            })

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }

    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved: {checkpoint_path}')


def train(args):
    """Main training function"""

    # load config
    config = load_config()

    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f'Using device: {device}')

    # get dataset info
    num_classes = config['datasets'][args.dataset]['num_classes']

    # load data
    print(f'\nLoading {args.dataset} dataset...')
    train_loader, val_loader, test_loader = get_dataloaders(args.dataset, config)

    # create model
    print(f'\nCreating {args.model} model...')
    model = get_model(args.model, num_classes=num_classes, pretrained=args.pretrained)
    model = model.to(device)
    print_model_summary(model, args.model)

    # setup training
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=config['training']['momentum'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=config['training']['weight_decay']
        )

    # learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # early stopping
    early_stopping = EarlyStopping(patience=args.patience)

    # checkpoint directory
    checkpoint_dir = os.path.join('checkpoints', args.dataset, args.model)
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')

    # training loop
    print(f'\nStarting training for {args.epochs} epochs...\n')

    best_val_acc = 0.0
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )

        # update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # epoch summary
        epoch_time = time.time() - epoch_start
        print(f'\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s):')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%')
        print(f'  LR: {current_lr:.6f}')

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, best_checkpoint_path)
            print(f'  New best validation accuracy: {best_val_acc:.2f}%')

        # check early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f'\nEarly stopping triggered at epoch {epoch}')
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
