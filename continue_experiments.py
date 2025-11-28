"""
Continue experiments from where they stopped.
Skips already completed experiments.
"""

import os
from run_experiments import MODELS, DATASETS, run_experiment

def check_completed(model, dataset):
    """Check if experiment is already completed."""
    checkpoint_path = f'checkpoints/{dataset}/{model}/best_model.pth'
    return os.path.exists(checkpoint_path)

def main():
    """Continue running experiments."""
    total_experiments = len(MODELS) * len(DATASETS)
    experiment_num = 0
    successful = 0
    failed = 0
    skipped = 0

    print("\n" + "="*80)
    print("CONTINUING WEEK 4 TRAINING EXPERIMENTS")
    print("Skipping completed experiments...")
    print("="*80 + "\n")

    for model in MODELS:
        for dataset in DATASETS:
            experiment_num += 1

            if check_completed(model, dataset):
                print(f"\nSkipping Experiment {experiment_num}/{total_experiments}")
                print(f"Model: {model} | Dataset: {dataset} - ALREADY COMPLETED")
                skipped += 1
                successful += 1
                continue

            success = run_experiment(model, dataset, experiment_num, total_experiments)

            if success:
                successful += 1
            else:
                failed += 1

    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETED")
    print(f"Successful: {successful}/{total_experiments}")
    print(f"Skipped (already done): {skipped}/{total_experiments}")
    print(f"Failed: {failed}/{total_experiments}")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
