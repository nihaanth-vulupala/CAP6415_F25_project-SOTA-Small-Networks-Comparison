"""
Monitor training progress.
Shows which experiments have completed and current progress.
"""

import os
import glob
import torch
from datetime import datetime

def check_experiment_status():
    """Check status of all 12 experiments."""
    models = ['mobilenetv3', 'efficientnet', 'shufflenet', 'squeezenet']
    datasets = ['cifar100', 'stanford_dogs', 'flowers102']

    total_experiments = len(models) * len(datasets)
    completed = 0
    pending = 0

    print("\n" + "="*80)
    print("EXPERIMENT PROGRESS MONITOR")
    print("="*80)
    print(f"{'Model':<15} {'Dataset':<15} {'Status':<10} {'Val Acc':<10} {'Epoch':<8}")
    print("-"*80)

    for model in models:
        for dataset in datasets:
            checkpoint_path = f'checkpoints/{dataset}/{model}/best_model.pth'

            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                val_acc = checkpoint.get('val_accuracy', 0.0)
                epoch = checkpoint.get('epoch', 0)
                status = "DONE"
                completed += 1

                print(f"{model:<15} {dataset:<15} {status:<10} {val_acc:<10.2f} {epoch:<8}")
            else:
                status = "PENDING"
                pending += 1
                print(f"{model:<15} {dataset:<15} {status:<10} {'--':<10} {'--':<8}")

    print("-"*80)
    print(f"Completed: {completed}/{total_experiments}")
    print(f"Pending: {pending}/{total_experiments}")
    print(f"Progress: {completed/total_experiments*100:.1f}%")
    print("="*80 + "\n")

def check_log_files():
    """Check recent log files for errors."""
    log_files = glob.glob('results/*_training.log')

    if not log_files:
        print("No training logs found yet.")
        return

    print("Recent training logs:")
    for log_file in sorted(log_files, key=os.path.getmtime, reverse=True)[:3]:
        file_size = os.path.getsize(log_file)
        mod_time = datetime.fromtimestamp(os.path.getmtime(log_file))
        print(f"  {os.path.basename(log_file)}: {file_size/1024:.1f} KB (modified: {mod_time.strftime('%H:%M:%S')})")

if __name__ == '__main__':
    check_experiment_status()
    check_log_files()
