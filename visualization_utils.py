import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List

def visualize_training_progress(metrics_tracker):
    fig = plt.figure(figsize=(20, 15))
    
    # Plot pattern-specific losses
    ax1 = plt.subplot(3, 2, 1)
    plot_pattern_losses(metrics_tracker.pattern_losses, ax1)
    
    # Plot attention evolution
    ax2 = plt.subplot(3, 2, 2)
    plot_attention_metrics(metrics_tracker.attention_stats, ax2)
    
    # Plot memory statistics
    ax3 = plt.subplot(3, 2, 3)
    plot_memory_stats(metrics_tracker.memory_stats, ax3)
    
    # Plot layer statistics
    ax4 = plt.subplot(3, 2, 4)
    plot_layer_stats(metrics_tracker.layer_stats, ax4)
    
    plt.tight_layout()
    plt.show()

def plot_pattern_losses(pattern_losses: Dict[str, List[float]], ax):
    df = pd.DataFrame(pattern_losses)
    sns.lineplot(data=df, ax=ax)
    ax.set_title('Pattern-Specific Learning Curves')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')

def plot_attention_metrics(attention_stats: Dict[str, List[float]], ax):
    df = pd.DataFrame(attention_stats)
    sns.lineplot(data=df, ax=ax)
    ax.set_title('Attention Metrics Evolution')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Metric Value')

def plot_memory_stats(memory_stats: Dict[str, List[float]], ax):
    df = pd.DataFrame(memory_stats)
    sns.lineplot(data=df, ax=ax)
    ax.set_title('Memory Statistics')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Metric Value')

def plot_layer_stats(layer_stats: Dict[str, List[Dict]], ax):
    # Reshape layer statistics for plotting
    data = []
    for layer_name, stats_list in layer_stats.items():
        for step, stats in enumerate(stats_list):
            for metric, value in stats.items():
                data.append({
                    'Layer': layer_name,
                    'Step': step,
                    'Metric': metric,
                    'Value': value
                })
    
    df = pd.DataFrame(data)
    sns.lineplot(
        data=df, x='Step', y='Value', 
        hue='Layer', style='Metric', ax=ax
    )
    ax.set_title('Layer-wise Statistics') 