"""
Experiment runner for training all model-dataset combinations.
Runs all 12 experiments systematically with proper logging.
"""

import subprocess
import os
import time
from datetime import datetime

# Configuration
MODELS = ['mobilenetv3', 'efficientnet', 'shufflenet', 'squeezenet']
DATASETS = ['cifar100', 'stanford_dogs', 'flowers102']
EPOCHS = 15
BATCH_SIZE = 128
LR = 0.01
PATIENCE = 5

# Results directory
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_experiment(model, dataset, experiment_num, total_experiments):
    """Run a single training experiment."""
    print(f"\n{'='*80}")
    print(f"Experiment {experiment_num}/{total_experiments}")
    print(f"Model: {model} | Dataset: {dataset}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    # Create log file
    log_file = os.path.join(RESULTS_DIR, f"{model}_{dataset}_training.log")

    # Build command
    cmd = [
        '.venv/bin/python', 'train.py',
        '--model', model,
        '--dataset', dataset,
        '--epochs', str(EPOCHS),
        '--lr', str(LR),
        '--patience', str(PATIENCE),
        '--pretrained'
    ]

    # Run training
    start_time = time.time()

    try:
        with open(log_file, 'w') as log:
            log.write(f"Experiment: {model} on {dataset}\n")
            log.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"Command: {' '.join(cmd)}\n\n")
            log.flush()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end='')
                log.write(line)
                log.flush()

            process.wait()

        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)

        print(f"\n{'='*80}")
        print(f"Experiment {experiment_num}/{total_experiments} completed")
        print(f"Time taken: {hours}h {minutes}m")
        print(f"Log saved: {log_file}")
        print(f"{'='*80}\n")

        return True

    except Exception as e:
        print(f"\nError running experiment: {e}")
        return False

def main():
    """Run all experiments."""
    print("\n" + "="*80)
    print("WEEK 4 TRAINING EXPERIMENTS")
    print("Running all 12 model-dataset combinations")
    print("="*80)

    total_experiments = len(MODELS) * len(DATASETS)
    experiment_num = 0
    successful = 0
    failed = 0

    # Create summary file
    summary_file = os.path.join(RESULTS_DIR, 'experiment_summary.txt')

    with open(summary_file, 'w') as f:
        f.write("WEEK 4 EXPERIMENT SUMMARY\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total experiments: {total_experiments}\n\n")

    # Run all experiments
    for model in MODELS:
        for dataset in DATASETS:
            experiment_num += 1

            success = run_experiment(model, dataset, experiment_num, total_experiments)

            if success:
                successful += 1
            else:
                failed += 1

            # Update summary
            with open(summary_file, 'a') as f:
                status = "SUCCESS" if success else "FAILED"
                f.write(f"{experiment_num}. {model} on {dataset}: {status}\n")

    # Final summary
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print(f"Successful: {successful}/{total_experiments}")
    print(f"Failed: {failed}/{total_experiments}")
    print(f"Summary saved: {summary_file}")
    print("="*80 + "\n")

    with open(summary_file, 'a') as f:
        f.write(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Successful: {successful}/{total_experiments}\n")
        f.write(f"Failed: {failed}/{total_experiments}\n")

if __name__ == '__main__':
    main()
