"""
Benchmark script to evaluate the performance of Liquid Neural Networks (LNNs)
and Digital Neocortex on pattern recognition tasks.

This script implements several pattern recognition tasks:
1. Sequence Replication: Replicate a given sequence
2. Pattern Completion: Complete a partially observed pattern
3. Temporal Pattern Recognition: Identify patterns that span across time
4. Noisy Pattern Recognition: Recognize patterns with high noise levels
5. Long-Range Dependency: Detect dependencies between distant time steps
6. Multi-Scale Pattern Recognition: Identify patterns at different time scales
7. Pattern Morphing: Recognize patterns that gradually transform over time
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse
import os

# Import our models
from liquid_neural_network import LiquidNeuralNetwork, DynamicWeightLNN
from digital_neocortex_components import DigitalNeocortex

# Enable PyTorch anomaly detection to find the exact operation causing the error
torch.autograd.set_detect_anomaly(True)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class PatternDataset:
    """
    Dataset generator for various pattern recognition tasks.
    """
    def __init__(self, task_type='replication', seq_len=20, input_dim=10, 
                 batch_size=32, num_patterns=5, noise_level=0.1):
        """
        Initialize the dataset generator.
        
        Args:
            task_type: Type of task ('replication', 'completion', 'temporal', 'noisy', 'long_range', 'multi_scale', 'morphing')
            seq_len: Length of sequences
            input_dim: Dimension of input features
            batch_size: Batch size for training/testing
            num_patterns: Number of distinct patterns to generate
            noise_level: Amount of noise to add to patterns
        """
        self.task_type = task_type
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.num_patterns = num_patterns
        self.noise_level = noise_level
        
        # Generate base patterns
        self.patterns = self._generate_base_patterns()
    
    def _generate_base_patterns(self):
        """Generate a set of distinct base patterns"""
        patterns = []
        
        for _ in range(self.num_patterns):
            if self.task_type == 'temporal':
                # For temporal tasks, create patterns that evolve over time
                pattern = np.zeros((self.seq_len, self.input_dim))
                
                # Create a few random "motifs" that repeat
                motif_length = min(5, self.seq_len // 2)
                motif = np.random.randn(motif_length, self.input_dim)
                
                # Place motifs at random positions
                for i in range(0, self.seq_len - motif_length, motif_length):
                    if np.random.rand() > 0.3:  # 70% chance to place a motif
                        pattern[i:i+motif_length] = motif
                
                # Normalize
                pattern = pattern / np.max(np.abs(pattern))
            elif self.task_type == 'noisy':
                # For noisy tasks, create patterns with high noise levels
                pattern = np.random.randn(self.seq_len, self.input_dim)
                pattern = pattern / np.max(np.abs(pattern))
            elif self.task_type == 'long_range':
                # For long-range dependency tasks, create patterns with dependencies between distant time steps
                pattern = np.zeros((self.seq_len, self.input_dim))
                
                # Create a few random "motifs" that repeat
                motif_length = min(5, self.seq_len // 2)
                motif = np.random.randn(motif_length, self.input_dim)
                
                # Place motifs at random positions
                for i in range(0, self.seq_len - motif_length, motif_length):
                    if np.random.rand() > 0.3:  # 70% chance to place a motif
                        pattern[i:i+motif_length] = motif
                
                # Add dependencies between distant time steps
                for i in range(self.seq_len):
                    if np.random.rand() > 0.5:
                        pattern[i] += pattern[(i + 10) % self.seq_len]
                
                # Normalize
                pattern = pattern / np.max(np.abs(pattern))
            elif self.task_type == 'multi_scale':
                # For multi-scale tasks, create patterns at different time scales
                pattern = np.zeros((self.seq_len, self.input_dim))
                
                # Create a few random "motifs" that repeat
                motif_length = min(5, self.seq_len // 2)
                motif = np.random.randn(motif_length, self.input_dim)
                
                # Place motifs at random positions
                for i in range(0, self.seq_len - motif_length, motif_length):
                    if np.random.rand() > 0.3:  # 70% chance to place a motif
                        pattern[i:i+motif_length] = motif
                
                # Add patterns at different time scales
                for i in range(self.seq_len):
                    if np.random.rand() > 0.5:
                        pattern[i] += pattern[i // 2]
                
                # Normalize
                pattern = pattern / np.max(np.abs(pattern))
            elif self.task_type == 'morphing':
                # For morphing tasks, create patterns that gradually transform over time
                pattern = np.zeros((self.seq_len, self.input_dim))
                
                # Create start and end patterns
                start_pattern = np.random.randn(self.input_dim)
                end_pattern = np.random.randn(self.input_dim)
                
                # Create a smooth transition between patterns
                for i in range(self.seq_len):
                    # Calculate morphing factor (0 to 1)
                    morph_factor = i / (self.seq_len - 1)
                    # Linear interpolation between start and end patterns
                    pattern[i] = (1 - morph_factor) * start_pattern + morph_factor * end_pattern
                
                # Add some temporal structure with oscillations
                temporal_freq = np.random.uniform(0.1, 0.3)  # Random frequency
                temporal_mod = np.sin(np.arange(self.seq_len) * temporal_freq * 2 * np.pi)
                for i in range(self.input_dim):
                    pattern[:, i] += temporal_mod * np.random.uniform(0.1, 0.5)
                
                # Normalize
                pattern = pattern / np.max(np.abs(pattern))
            else:
                # For other tasks, create static patterns
                pattern = np.random.randn(self.input_dim)
                pattern = pattern / np.max(np.abs(pattern))
            
            patterns.append(pattern)
        
        return patterns
    
    def generate_batch(self, split='train'):
        """
        Generate a batch of data for the specified task.
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            inputs: Input tensor of shape (batch_size, seq_len, input_dim)
            targets: Target tensor of shape (batch_size, seq_len, input_dim)
            pattern_ids: IDs of patterns used in this batch
        """
        inputs = []
        targets = []
        pattern_ids = []
        
        # Add more noise to test set
        noise_level = self.noise_level if split == 'train' else self.noise_level * 2
        
        for _ in range(self.batch_size):
            # Select a random pattern
            pattern_id = np.random.randint(0, self.num_patterns)
            pattern = self.patterns[pattern_id]
            pattern_ids.append(pattern_id)
            
            if self.task_type == 'replication':
                # Task: Replicate the pattern exactly
                if isinstance(pattern, np.ndarray) and pattern.ndim == 1:
                    # Expand static pattern across time
                    target = np.tile(pattern, (self.seq_len, 1))
                else:
                    # Use temporal pattern as is
                    target = pattern
                
                # Add noise to input
                input_seq = target + np.random.randn(*target.shape) * noise_level
                
            elif self.task_type == 'completion':
                # Task: Complete a partially observed pattern
                if isinstance(pattern, np.ndarray) and pattern.ndim == 1:
                    # Expand static pattern across time
                    target = np.tile(pattern, (self.seq_len, 1))
                else:
                    # Use temporal pattern as is
                    target = pattern
                
                # Create input with masked portions
                input_seq = target.copy()
                
                # Mask random portions (set to zero)
                mask = np.random.rand(self.seq_len, self.input_dim) > 0.3
                input_seq = input_seq * mask
                
                # Add noise to non-masked portions
                input_seq = input_seq + np.random.randn(*input_seq.shape) * noise_level * mask
                
            elif self.task_type == 'temporal':
                # Task: Recognize temporal patterns
                target = pattern.copy()
                
                # Add noise and temporal shifts
                input_seq = target + np.random.randn(*target.shape) * noise_level
                
                # Randomly shift some sequences in time
                if np.random.rand() > 0.5:
                    shift = np.random.randint(1, 3)
                    input_seq = np.roll(input_seq, shift, axis=0)
            
            elif self.task_type == 'noisy':
                # Task: Recognize patterns with high noise levels
                target = pattern.copy()
                
                # Add high noise levels
                input_seq = target + np.random.randn(*target.shape) * 0.5
            
            elif self.task_type == 'long_range':
                # Task: Detect dependencies between distant time steps
                target = pattern.copy()
                
                # Add dependencies between distant time steps
                for i in range(self.seq_len):
                    if np.random.rand() > 0.5:
                        input_seq[i] += input_seq[(i + 10) % self.seq_len]
            
            elif self.task_type == 'multi_scale':
                # Task: Identify patterns at different time scales
                target = pattern.copy()
                
                # Add patterns at different time scales
                for i in range(self.seq_len):
                    if np.random.rand() > 0.5:
                        input_seq[i] += input_seq[i // 2]
            
            elif self.task_type == 'morphing':
                # Task: Recognize patterns that gradually transform over time
                target = pattern.copy()
                
                # Add noise to input that increases over time
                noise = np.random.randn(*target.shape) * noise_level
                # Make noise level increase with time
                noise_scale = np.linspace(0.5, 2.0, self.seq_len).reshape(-1, 1)
                noise = noise * noise_scale
                
                # Create input with increasing noise
                input_seq = target + noise
                
                # Randomly mask some time steps to make it more challenging
                mask = np.random.rand(self.seq_len) > 0.2  # 20% chance to mask
                mask = mask.reshape(-1, 1)
                input_seq = input_seq * mask
                
            inputs.append(input_seq)
            targets.append(target)
        
        # Convert to PyTorch tensors
        inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
        targets = torch.tensor(np.array(targets), dtype=torch.float32)
        
        return inputs, targets, pattern_ids


def train_model(model, dataset, num_epochs=50, lr=0.001, device='cuda'):
    """
    Train a model on the given dataset.
    
    Args:
        model: Model to train
        dataset: PatternDataset instance
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to use for training
        
    Returns:
        train_losses: List of training losses
        test_losses: List of testing losses
    """
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for _ in range(10):  # 10 batches per epoch
            # Generate batch
            inputs, targets, _ = dataset.generate_batch(split='train')
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Handle different return types (tuple or tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Unpack tuple, only use outputs
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= 10
        train_losses.append(train_loss)
        
        # Testing
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for _ in range(5):  # 5 test batches
                # Generate batch
                inputs, targets, _ = dataset.generate_batch(split='test')
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Handle different return types (tuple or tensor)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Unpack tuple, only use outputs
                
                # Compute loss
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        
        test_loss /= 5
        test_losses.append(test_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    return train_losses, test_losses


def evaluate_model(model, dataset, num_batches=10, device='cuda'):
    """
    Evaluate a model on the given dataset.
    
    Args:
        model: Model to evaluate
        dataset: PatternDataset instance
        num_batches: Number of batches to evaluate on
        device: Device to use for evaluation
        
    Returns:
        mse: Mean squared error
        accuracy: Pattern recognition accuracy
        inference_time: Average inference time per batch
    """
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Evaluation metrics
    mse_total = 0.0
    correct = 0
    total = 0
    inference_time = 0.0
    
    with torch.no_grad():
        for _ in range(num_batches):
            # Generate batch
            inputs, targets, pattern_ids = dataset.generate_batch(split='test')
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(inputs)
            inference_time += time.time() - start_time
            
            # Handle different return types (tuple or tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Unpack tuple, only use outputs
            
            # Compute MSE
            mse = torch.mean((outputs - targets) ** 2).item()
            mse_total += mse
            
            # Compute pattern recognition accuracy
            if dataset.task_type == 'replication' or dataset.task_type == 'completion':
                # For replication/completion, check if the output pattern is closest to the target pattern
                for i in range(outputs.shape[0]):
                    output_pattern = outputs[i].mean(dim=0)
                    target_pattern = targets[i].mean(dim=0)
                    
                    # Find closest pattern
                    min_dist = float('inf')
                    pred_pattern = -1
                    
                    for j, pattern in enumerate(dataset.patterns):
                        if isinstance(pattern, np.ndarray) and pattern.ndim == 1:
                            pattern_tensor = torch.tensor(pattern, device=device)
                            dist = torch.mean((output_pattern - pattern_tensor) ** 2).item()
                            
                            if dist < min_dist:
                                min_dist = dist
                                pred_pattern = j
                    
                    if pred_pattern == pattern_ids[i]:
                        correct += 1
                    
                    total += 1
            
            elif dataset.task_type == 'temporal':
                # For temporal tasks, check if the output sequence matches the target sequence
                for i in range(outputs.shape[0]):
                    output_seq = outputs[i]
                    target_seq = targets[i]
                    
                    # Compute sequence similarity
                    similarity = torch.mean((output_seq - target_seq) ** 2, dim=1)
                    
                    # If similarity is below threshold for most time steps, count as correct
                    if torch.mean(similarity < 0.2).float().item() > 0.7:
                        correct += 1
                    
                    total += 1
            
            elif dataset.task_type == 'noisy':
                # For noisy tasks, check if the output sequence matches the target sequence
                for i in range(outputs.shape[0]):
                    output_seq = outputs[i]
                    target_seq = targets[i]
                    
                    # Compute sequence similarity
                    similarity = torch.mean((output_seq - target_seq) ** 2, dim=1)
                    
                    # If similarity is below threshold for most time steps, count as correct
                    if torch.mean(similarity < 0.2).float().item() > 0.7:
                        correct += 1
                    
                    total += 1
            
            elif dataset.task_type == 'long_range':
                # For long-range dependency tasks, check if the output sequence matches the target sequence
                for i in range(outputs.shape[0]):
                    output_seq = outputs[i]
                    target_seq = targets[i]
                    
                    # Compute sequence similarity
                    similarity = torch.mean((output_seq - target_seq) ** 2, dim=1)
                    
                    # If similarity is below threshold for most time steps, count as correct
                    if torch.mean(similarity < 0.2).float().item() > 0.7:
                        correct += 1
                    
                    total += 1
            
            elif dataset.task_type == 'multi_scale':
                # For multi-scale tasks, check if the output sequence matches the target sequence
                for i in range(outputs.shape[0]):
                    output_seq = outputs[i]
                    target_seq = targets[i]
                    
                    # Compute sequence similarity
                    similarity = torch.mean((output_seq - target_seq) ** 2, dim=1)
                    
                    # If similarity is below threshold for most time steps, count as correct
                    if torch.mean(similarity < 0.2).float().item() > 0.7:
                        correct += 1
                    
                    total += 1
            
            elif dataset.task_type == 'morphing':
                # For morphing tasks, check if the output sequence matches the target sequence
                for i in range(outputs.shape[0]):
                    output_seq = outputs[i]
                    target_seq = targets[i]
                    
                    # Compute sequence similarity
                    similarity = torch.mean((output_seq - target_seq) ** 2, dim=1)
                    
                    # If similarity is below threshold for most time steps, count as correct
                    if torch.mean((similarity < 0.2).float()).item() > 0.7:
                        correct += 1
                    
                    total += 1
    
    # Compute average metrics
    mse = mse_total / num_batches
    accuracy = correct / total if total > 0 else 0.0
    inference_time = inference_time / num_batches
    
    return mse, accuracy, inference_time


def visualize_pattern_replication(model, dataset, device='cuda', save_path='results'):
    """
    Visualize how well a model replicates patterns.
    
    Args:
        model: Model to evaluate
        dataset: PatternDataset instance
        device: Device to use for evaluation
        save_path: Directory to save visualizations
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate a batch of data
    inputs, targets, pattern_ids = dataset.generate_batch(split='test')
    inputs, targets = inputs.to(device), targets.to(device)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Unpack tuple, only use outputs
    
    # Convert to numpy for plotting
    inputs = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    outputs = outputs.cpu().numpy()
    
    # Get model name
    model_name = model.__class__.__name__
    
    # Create a single comparison figure for the first pattern
    plt.figure(figsize=(15, 5))
    
    # Plot input pattern
    plt.subplot(1, 3, 1)
    if inputs.shape[1] > 1:  # Sequence data
        plt.imshow(inputs[0].T, aspect='auto', cmap='viridis')
        plt.title('Input Pattern')
        plt.xlabel('Time Step')
        plt.ylabel('Feature Dimension')
    else:  # Static pattern
        plt.bar(range(inputs.shape[2]), inputs[0, 0])
        plt.title('Input Pattern')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Value')
    
    # Plot target pattern
    plt.subplot(1, 3, 2)
    if targets.shape[1] > 1:  # Sequence data
        plt.imshow(targets[0].T, aspect='auto', cmap='viridis')
        plt.title('Target Pattern')
        plt.xlabel('Time Step')
        plt.ylabel('Feature Dimension')
    else:  # Static pattern
        plt.bar(range(targets.shape[2]), targets[0, 0])
        plt.title('Target Pattern')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Value')
    
    # Plot model output
    plt.subplot(1, 3, 3)
    if outputs.shape[1] > 1:  # Sequence data
        plt.imshow(outputs[0].T, aspect='auto', cmap='viridis')
        plt.title(f'{model_name} Output')
        plt.xlabel('Time Step')
        plt.ylabel('Feature Dimension')
    else:  # Static pattern
        plt.bar(range(outputs.shape[2]), outputs[0, 0])
        plt.title(f'{model_name} Output')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/{model_name}_replication.png')
    plt.close()
    
    print(f"Visualization saved to {save_path}/{model_name}_replication.png")


def run_benchmark(task_type='replication', seq_len=20, input_dim=10, hidden_dim=64,
                  batch_size=32, num_patterns=5, noise_level=0.1, num_epochs=50,
                  device='cuda'):
    """
    Run benchmark comparing LNN, DynamicWeightLNN, and DigitalNeocortex.
    
    Args:
        task_type: Type of pattern recognition task
        seq_len: Sequence length
        input_dim: Input dimension
        hidden_dim: Hidden dimension for models
        batch_size: Batch size
        num_patterns: Number of patterns to generate
        noise_level: Noise level for inputs
        num_epochs: Number of training epochs
        device: Device to use
        
    Returns:
        results: Dictionary of benchmark results
    """
    # Create dataset
    dataset = PatternDataset(
        task_type=task_type,
        seq_len=seq_len,
        input_dim=input_dim,
        batch_size=batch_size,
        num_patterns=num_patterns,
        noise_level=noise_level
    )
    
    # Create models
    models = {
        'LNN': LiquidNeuralNetwork(input_dim, hidden_dim, input_dim),
        'DynamicWeightLNN': DynamicWeightLNN(input_dim, hidden_dim, input_dim),
        'DigitalNeocortex': DigitalNeocortex(input_dim, hidden_dim, input_dim)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name} on {task_type} task...")
        
        # Train model
        train_losses, test_losses = train_model(
            model, dataset, num_epochs=num_epochs, device=device
        )
        
        # Evaluate model
        mse, accuracy, inference_time = evaluate_model(
            model, dataset, device=device
        )
        
        # Visualize pattern replication
        visualize_pattern_replication(model, dataset, device=device, save_path=f'results/{task_type}')
        
        # Store results
        results[name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'mse': mse,
            'accuracy': accuracy,
            'inference_time': inference_time
        }
        
        print(f"{name} Results:")
        print(f"  MSE: {mse:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Inference Time: {inference_time:.4f} seconds")
        print(f"  Final Training Loss: {train_losses[-1]:.4f}")
    
    return results


def plot_results(results, task_type, save_dir='results'):
    """
    Plot benchmark results.
    
    Args:
        results: Dictionary of benchmark results
        task_type: Type of pattern recognition task
        save_dir: Directory to save plots
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    
    for name, result in results.items():
        plt.plot(result['train_losses'], label=f"{name} (Train)")
        plt.plot(result['test_losses'], label=f"{name} (Test)", linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Curves for {task_type.capitalize()} Task')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/{task_type}_training_curves.png")
    
    # Plot performance metrics
    metrics = ['mse', 'accuracy', 'inference_time']
    titles = ['Mean Squared Error', 'Pattern Recognition Accuracy', 'Inference Time (seconds)']
    
    for metric, title in zip(metrics, titles):
        plt.figure(figsize=(8, 6))
        
        values = [result[metric] for result in results.values()]
        names = list(results.keys())
        
        plt.bar(names, values)
        plt.ylabel(title)
        plt.title(f'{title} for {task_type.capitalize()} Task')
        plt.grid(True, axis='y')
        
        # Add value labels
        for i, v in enumerate(values):
            plt.text(i, v, f"{v:.4f}", ha='center', va='bottom')
        
        plt.savefig(f"{save_dir}/{task_type}_{metric}.png")
    
    plt.close('all')


def main():
    """Main function to run benchmarks"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Benchmark LNN vs Digital Neocortex')
    parser.add_argument('--task', type=str, default='replication',
                        choices=['replication', 'completion', 'temporal', 'noisy', 'long_range', 'multi_scale', 'morphing'],
                        help='Pattern recognition task')
    parser.add_argument('--seq_len', type=int, default=20,
                        help='Sequence length')
    parser.add_argument('--input_dim', type=int, default=10,
                        help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_patterns', type=int, default=5,
                        help='Number of patterns')
    parser.add_argument('--noise_level', type=float, default=0.1,
                        help='Noise level')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead.")
        args.device = 'cpu'
    
    # Run benchmark
    results = run_benchmark(
        task_type=args.task,
        seq_len=args.seq_len,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        num_patterns=args.num_patterns,
        noise_level=args.noise_level,
        num_epochs=args.num_epochs,
        device=args.device
    )
    
    # Plot results
    plot_results(results, args.task, save_dir=args.save_dir)


if __name__ == '__main__':
    main()
