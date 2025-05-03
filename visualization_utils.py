import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

class NeocortexVisualizer:
    def __init__(self):
        self.fig_size = (12, 8)
    
    def plot_attention_patterns(self, attention_weights):
        """Visualize attention patterns"""
        plt.figure(figsize=self.fig_size)
        sns.heatmap(attention_weights.detach().numpy(), 
                   cmap='viridis',
                   xticklabels=False,
                   yticklabels=False)
        plt.title('Attention Patterns')
        plt.show()
    
    def plot_memory_access(self, memory_weights):
        """Visualize memory access patterns"""
        plt.figure(figsize=self.fig_size)
        plt.plot(memory_weights.detach().numpy())
        plt.title('Memory Access Patterns')
        plt.xlabel('Memory Location')
        plt.ylabel('Access Weight')
        plt.show()
    
    def plot_processing_steps(self, outputs_dict):
        """Visualize processing steps"""
        num_steps = len(outputs_dict)
        fig, axes = plt.subplots(1, num_steps, figsize=(15, 5))
        
        for i, (name, tensor) in enumerate(outputs_dict.items()):
            if isinstance(tensor, torch.Tensor):
                sns.heatmap(tensor.detach().numpy()[:10, :10], 
                           ax=axes[i],
                           cmap='viridis')
                axes[i].set_title(name)
        
        plt.tight_layout()
        plt.show()

def visualize_inference(model_outputs, visualizer):
    """Visualize the inference results"""
    # Plot attention patterns if available
    if 'attention' in model_outputs:
        visualizer.plot_attention_patterns(model_outputs['attention'])
    
    # Plot memory access if available
    if 'memory_access' in model_outputs:
        visualizer.plot_memory_access(model_outputs['memory_access'])
    
    # Plot processing steps
    visualizer.plot_processing_steps(model_outputs) 