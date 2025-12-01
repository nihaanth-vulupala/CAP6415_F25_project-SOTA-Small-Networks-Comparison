"""
Create Additional Visualizations for Demo Presentation

Generates comprehensive comparison charts for video demonstration:
- Model architecture comparison
- Performance heatmap across datasets
- Efficiency frontier analysis
- Deployment recommendations
- Training insights

These visualizations complement the existing evaluation charts and provide
clear insights for the video demo presentation.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_results():
    """Load evaluation results from JSON file"""
    with open('results/evaluation_results.json', 'r') as f:
        results = json.load(f)
    return pd.DataFrame(results)

def create_architecture_comparison(df):
    """
    Create bar chart comparing model architectures
    Shows parameters and model size side-by-side
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Get unique models and their stats
    models_data = df.groupby('model').first()[['total_params', 'model_size_mb']]
    models = models_data.index.tolist()

    # Convert parameter counts to millions
    params_millions = models_data['total_params'] / 1_000_000
    sizes_mb = models_data['model_size_mb']

    # Color palette
    colors = sns.color_palette("husl", len(models))

    # Plot 1: Parameter counts
    bars1 = ax1.barh(models, params_millions, color=colors)
    ax1.set_xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Size: Parameter Count', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar in bars1:
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}M', ha='left', va='center', fontsize=10, fontweight='bold')

    # Plot 2: Model sizes in MB
    bars2 = ax2.barh(models, sizes_mb, color=colors)
    ax2.set_xlabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Size: Memory Footprint', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar in bars2:
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.2f} MB', ha='left', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/figures/architecture_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/architecture_comparison.png")
    plt.close()

def create_performance_heatmap(df):
    """
    Create heatmap showing model performance across all datasets
    Color-coded by accuracy
    """
    # Pivot table for heatmap
    heatmap_data = df.pivot(index='model', columns='dataset', values='test_accuracy')

    # Create figure
    plt.figure(figsize=(10, 6))

    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=100, cbar_kws={'label': 'Test Accuracy (%)'})

    plt.title('Model Performance Heatmap: Test Accuracy Across Datasets',
             fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Dataset', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/figures/performance_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/performance_heatmap.png")
    plt.close()

def create_efficiency_frontier(df):
    """
    Create scatter plot showing efficiency frontier
    Accuracy vs Inference Time with model size as bubble size
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    datasets = df['dataset'].unique()

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        dataset_df = df[df['dataset'] == dataset]

        # Create scatter plot
        for model in dataset_df['model'].unique():
            model_data = dataset_df[dataset_df['model'] == model]

            ax.scatter(model_data['avg_inference_time_ms'],
                      model_data['test_accuracy'],
                      s=model_data['model_size_mb'] * 20,  # Bubble size
                      alpha=0.6,
                      label=model)

            # Add model name annotations
            ax.annotate(model,
                       (model_data['avg_inference_time_ms'].values[0],
                        model_data['test_accuracy'].values[0]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax.set_xlabel('Inference Time (ms/batch)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{dataset.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    plt.suptitle('Efficiency Frontier: Accuracy vs Speed (Bubble size = Model Size)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/figures/efficiency_frontier.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/efficiency_frontier.png")
    plt.close()

def create_deployment_recommendations(df):
    """
    Create visualization showing best model for each deployment scenario
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Scenario 1: Best accuracy (top-left)
    ax = axes[0, 0]
    best_acc = df.groupby('model')['test_accuracy'].mean().sort_values(ascending=False)
    colors_acc = ['green' if i == 0 else 'lightblue' for i in range(len(best_acc))]
    best_acc.plot(kind='barh', ax=ax, color=colors_acc)
    ax.set_xlabel('Average Test Accuracy (%)', fontweight='bold')
    ax.set_title('Best for Accuracy-Critical Applications', fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)

    # Scenario 2: Smallest size (top-right)
    ax = axes[0, 1]
    smallest = df.groupby('model')['model_size_mb'].first().sort_values()
    colors_size = ['green' if i == 0 else 'lightcoral' for i in range(len(smallest))]
    smallest.plot(kind='barh', ax=ax, color=colors_size)
    ax.set_xlabel('Model Size (MB)', fontweight='bold')
    ax.set_title('Best for Size-Constrained Devices', fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)

    # Scenario 3: Fastest inference (bottom-left)
    ax = axes[1, 0]
    fastest = df.groupby('model')['avg_inference_time_ms'].mean().sort_values()
    colors_speed = ['green' if i == 0 else 'lightyellow' for i in range(len(fastest))]
    fastest.plot(kind='barh', ax=ax, color=colors_speed)
    ax.set_xlabel('Average Inference Time (ms)', fontweight='bold')
    ax.set_title('Best for Real-Time Applications', fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)

    # Scenario 4: Best balance (bottom-right)
    ax = axes[1, 1]
    # Calculate efficiency score: (accuracy / 100) / (size_mb * inference_time)
    df_balance = df.groupby('model').apply(
        lambda x: (x['test_accuracy'].mean() / 100) /
                 (x['model_size_mb'].mean() * x['avg_inference_time_ms'].mean() / 1000)
    ).sort_values(ascending=False)
    colors_balance = ['green' if i == 0 else 'lightgreen' for i in range(len(df_balance))]
    df_balance.plot(kind='barh', ax=ax, color=colors_balance)
    ax.set_xlabel('Efficiency Score (Higher is Better)', fontweight='bold')
    ax.set_title('Best Overall Balance', fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)

    plt.suptitle('Deployment Recommendations by Use Case', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('results/figures/deployment_recommendations.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/deployment_recommendations.png")
    plt.close()

def create_dataset_difficulty_analysis(df):
    """
    Analyze and visualize dataset difficulty based on model performance
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Average accuracy per dataset
    dataset_avg = df.groupby('dataset')['test_accuracy'].mean().sort_values()
    colors = ['red' if acc < 50 else 'orange' if acc < 70 else 'green' for acc in dataset_avg]

    ax1.barh(dataset_avg.index, dataset_avg.values, color=colors)
    ax1.set_xlabel('Average Test Accuracy (%)', fontweight='bold')
    ax1.set_title('Dataset Difficulty Ranking', fontweight='bold', fontsize=13)
    ax1.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (dataset, acc) in enumerate(dataset_avg.items()):
        ax1.text(acc, i, f' {acc:.1f}%', va='center', fontweight='bold')

    # Performance variance per dataset
    dataset_std = df.groupby('dataset')['test_accuracy'].std().sort_values(ascending=False)
    ax2.barh(dataset_std.index, dataset_std.values, color='skyblue')
    ax2.set_xlabel('Accuracy Std Dev (%)', fontweight='bold')
    ax2.set_title('Model Sensitivity by Dataset', fontweight='bold', fontsize=13)
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (dataset, std) in enumerate(dataset_std.items()):
        ax2.text(std, i, f' {std:.1f}%', va='center', fontweight='bold')

    plt.suptitle('Dataset Analysis: Difficulty and Model Sensitivity',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('results/figures/dataset_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/dataset_analysis.png")
    plt.close()

def create_summary_dashboard(df):
    """
    Create comprehensive summary dashboard with key insights
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Overall best models
    ax1 = fig.add_subplot(gs[0, :2])
    overall_scores = df.groupby('model')['test_accuracy'].mean().sort_values(ascending=False)
    bars = ax1.bar(overall_scores.index, overall_scores.values,
                   color=sns.color_palette("viridis", len(overall_scores)))
    ax1.set_ylabel('Average Accuracy (%)', fontweight='bold')
    ax1.set_title('Overall Model Ranking by Average Accuracy', fontweight='bold', fontsize=13)
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 2. Key statistics table
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    stats_text = "KEY STATISTICS\n" + "="*30 + "\n\n"
    stats_text += f"Total Experiments: {len(df)}\n"
    stats_text += f"Models Compared: {df['model'].nunique()}\n"
    stats_text += f"Datasets Used: {df['dataset'].nunique()}\n\n"
    stats_text += f"Best Accuracy:\n{df.loc[df['test_accuracy'].idxmax(), 'model']}\n"
    stats_text += f"{df['test_accuracy'].max():.2f}% on\n{df.loc[df['test_accuracy'].idxmax(), 'dataset']}\n\n"
    stats_text += f"Smallest Model:\n{df.loc[df['model_size_mb'].idxmin(), 'model']}\n"
    stats_text += f"{df['model_size_mb'].min():.2f} MB\n\n"
    stats_text += f"Fastest Inference:\n{df.loc[df['avg_inference_time_ms'].idxmin(), 'model']}\n"
    stats_text += f"{df['avg_inference_time_ms'].min():.2f} ms"

    ax2.text(0.1, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. Model size comparison
    ax3 = fig.add_subplot(gs[1, 0])
    size_data = df.groupby('model')['model_size_mb'].first().sort_values()
    ax3.pie(size_data.values, labels=size_data.index, autopct='%1.1f%%',
           colors=sns.color_palette("pastel"))
    ax3.set_title('Model Size Distribution', fontweight='bold')

    # 4. Accuracy by dataset
    ax4 = fig.add_subplot(gs[1, 1:])
    dataset_perf = df.pivot_table(values='test_accuracy', index='model',
                                   columns='dataset', aggfunc='mean')
    dataset_perf.plot(kind='bar', ax=ax4, width=0.8)
    ax4.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax4.set_xlabel('Model', fontweight='bold')
    ax4.set_title('Model Performance Across Datasets', fontweight='bold', fontsize=13)
    ax4.legend(title='Dataset', fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')

    # 5. Speed vs Accuracy scatter
    ax5 = fig.add_subplot(gs[2, :2])
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        ax5.scatter(model_df['avg_inference_time_ms'], model_df['test_accuracy'],
                   s=100, alpha=0.6, label=model)
    ax5.set_xlabel('Inference Time (ms)', fontweight='bold')
    ax5.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax5.set_title('Speed-Accuracy Tradeoff', fontweight='bold', fontsize=13)
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)

    # 6. Parameter efficiency
    ax6 = fig.add_subplot(gs[2, 2])
    params_eff = df.groupby('model').apply(
        lambda x: x['test_accuracy'].mean() / (x['total_params'].mean() / 1e6)
    ).sort_values(ascending=False)
    ax6.barh(params_eff.index, params_eff.values, color='coral')
    ax6.set_xlabel('Accuracy per Million Params', fontweight='bold', fontsize=9)
    ax6.set_title('Parameter Efficiency', fontweight='bold', fontsize=11)
    ax6.grid(axis='x', alpha=0.3)

    plt.suptitle('SOTA Small Networks Comparison - Summary Dashboard',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('results/figures/summary_dashboard.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/summary_dashboard.png")
    plt.close()

def main():
    """Generate all demo visualizations"""
    print("Loading evaluation results...")
    df = load_results()

    print("\nGenerating demo visualizations...")
    print("-" * 50)

    create_architecture_comparison(df)
    create_performance_heatmap(df)
    create_efficiency_frontier(df)
    create_deployment_recommendations(df)
    create_dataset_difficulty_analysis(df)
    create_summary_dashboard(df)

    print("-" * 50)
    print("\nAll demo visualizations generated successfully!")
    print("Total figures created: 6 new visualizations")
    print("Check results/figures/ directory for output files.")

if __name__ == '__main__':
    main()
