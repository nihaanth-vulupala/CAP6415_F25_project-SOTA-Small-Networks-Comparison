"""
Visualization script for model comparison.
Generates charts comparing accuracy, model size, and inference time.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_results(results_file='results/evaluation_results.json'):
    """Load evaluation results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def plot_accuracy_comparison(results, output_dir='results/figures'):
    """Plot test accuracy comparison across models and datasets."""
    os.makedirs(output_dir, exist_ok=True)

    datasets = list(set([r['dataset'] for r in results]))
    models = list(set([r['model'] for r in results]))

    fig, axes = plt.subplots(1, len(datasets), figsize=(15, 5))
    if len(datasets) == 1:
        axes = [axes]

    for idx, dataset in enumerate(datasets):
        dataset_results = [r for r in results if r['dataset'] == dataset]

        model_names = [r['model'] for r in dataset_results]
        test_accs = [r['test_accuracy'] for r in dataset_results]

        axes[idx].bar(model_names, test_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[idx].set_title(f'{dataset.upper().replace("_", " ")}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Test Accuracy (%)', fontsize=10)
        axes[idx].set_ylim([0, 100])
        axes[idx].grid(axis='y', alpha=0.3)

        for i, acc in enumerate(test_accs):
            axes[idx].text(i, acc + 1, f'{acc:.1f}%', ha='center', fontsize=9)

        axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/accuracy_comparison.png")
    plt.close()

def plot_efficiency_comparison(results, output_dir='results/figures'):
    """Plot model size vs accuracy."""
    os.makedirs(output_dir, exist_ok=True)

    datasets = list(set([r['dataset'] for r in results]))

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'mobilenetv3': '#1f77b4', 'efficientnet': '#ff7f0e',
              'shufflenet': '#2ca02c', 'squeezenet': '#d62728'}

    for dataset in datasets:
        dataset_results = [r for r in results if r['dataset'] == dataset]

        for r in dataset_results:
            marker = 'o' if dataset == 'cifar100' else ('s' if dataset == 'stanford_dogs' else '^')
            ax.scatter(r['model_size_mb'], r['test_accuracy'],
                      color=colors.get(r['model'], '#333333'),
                      marker=marker, s=150, alpha=0.7,
                      label=f"{r['model']} ({dataset})" if dataset == 'cifar100' else '')

    ax.set_xlabel('Model Size (MB)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Model Efficiency: Size vs Accuracy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/efficiency_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/efficiency_comparison.png")
    plt.close()

def plot_inference_time(results, output_dir='results/figures'):
    """Plot inference time comparison."""
    os.makedirs(output_dir, exist_ok=True)

    models = sorted(list(set([r['model'] for r in results])))
    datasets = sorted(list(set([r['dataset'] for r in results])))

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, dataset in enumerate(datasets):
        times = []
        for model in models:
            model_result = [r for r in results if r['model'] == model and r['dataset'] == dataset]
            if model_result:
                times.append(model_result[0]['avg_inference_time_ms'])
            else:
                times.append(0)

        offset = width * (i - 1)
        ax.bar(x + offset, times, width, label=dataset.replace('_', ' ').title())

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Inference Time (ms/batch)', fontsize=12)
    ax.set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/inference_time_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/inference_time_comparison.png")
    plt.close()

def plot_model_comparison_table(results, output_dir='results/figures'):
    """Create a visual comparison table."""
    os.makedirs(output_dir, exist_ok=True)

    models = sorted(list(set([r['model'] for r in results])))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    table_data = [['Model', 'Dataset', 'Test Acc (%)', 'Params', 'Size (MB)', 'Time (ms)']]

    for model in models:
        model_results = [r for r in results if r['model'] == model]
        for r in sorted(model_results, key=lambda x: x['dataset']):
            table_data.append([
                r['model'],
                r['dataset'].replace('_', ' ').title(),
                f"{r['test_accuracy']:.2f}",
                f"{r['total_params']:,}",
                f"{r['model_size_mb']:.2f}",
                f"{r['avg_inference_time_ms']:.2f}"
            ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.2, 0.15, 0.15, 0.15, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')

    plt.title('Model Comparison Summary', fontsize=16, fontweight='bold', pad=20)

    plt.savefig(f'{output_dir}/comparison_table.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/comparison_table.png")
    plt.close()

def main():
    results_file = 'results/evaluation_results.json'

    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("Run evaluate.py first to generate results.")
        return

    print("Loading results...")
    results = load_results(results_file)

    print(f"Found {len(results)} evaluation results")

    print("\nGenerating visualizations...")
    plot_accuracy_comparison(results)
    plot_efficiency_comparison(results)
    plot_inference_time(results)
    plot_model_comparison_table(results)

    print("\nAll visualizations generated successfully!")
    print("Check results/figures/ directory for output files.")

if __name__ == '__main__':
    main()
