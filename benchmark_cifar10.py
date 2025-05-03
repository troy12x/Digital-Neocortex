import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
from tqdm import tqdm

from liquid_neural_network import LiquidNeuralNetwork, DynamicWeightLNN
from digital_neocortex_components import DigitalNeocortex

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class CIFARModel(nn.Module):
    """
    Base model for CIFAR classification.
    """
    def __init__(self, model_type, input_dim=3*32*32, hidden_dim=256, output_dim=10, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model_type = model_type
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Choose model type
        if model_type == 'lnn':
            self.network = LiquidNeuralNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                **kwargs
            )
        elif model_type == 'dynamic_weight_lnn':
            self.network = DynamicWeightLNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                **kwargs
            )
        elif model_type == 'digital_neocortex':
            # For Digital Neocortex, use a smaller hidden dimension to prevent overfitting
            neocortex_hidden_dim = kwargs.get('hidden_dim', hidden_dim)
            if 'hidden_dim' in kwargs:
                del kwargs['hidden_dim']
                
            self.network = DigitalNeocortex(
                input_dim=input_dim,
                hidden_dim=neocortex_hidden_dim,
                output_dim=neocortex_hidden_dim,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Output layer with L2 regularization
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights with smaller values for better generalization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with smaller values for better generalization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                # Check if parameter has at least 2 dimensions before applying Xavier init
                if len(param.shape) >= 2:
                    nn.init.xavier_normal_(param, gain=0.5)
                else:
                    # For 1D parameters (e.g., some bias terms that are named as weights)
                    nn.init.normal_(param, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        # Flatten the input
        x = self.flatten(x)
        
        # Apply dropout for regularization
        x = self.dropout(x)
        
        # Reshape for compatibility with LNN variants that expect 3D input
        # All our LNN variants expect 3D input: (batch_size, seq_len, features)
        x = x.unsqueeze(1)  # Add a sequence dimension (batch_size, 1, features)
        
        # Pass through the network
        x = self.network(x)
        
        # If the network returns a tuple (like some LNN variants), take the first element
        if isinstance(x, tuple):
            x = x[0]
        
        # If x has 3 dimensions (batch_size, seq_len, features), take the last sequence element
        if x.dim() == 3:
            x = x[:, -1, :]
        
        # Apply dropout again
        x = self.dropout(x)
        
        # Pass through the output layer
        x = self.output_layer(x)
        
        return x
    
    def reset_state(self):
        """Reset internal states"""
        if hasattr(self.network, 'reset_state'):
            self.network.reset_state()

def load_cifar_data(dataset_name='cifar10', batch_size=64, use_synthetic=False):
    """
    Load CIFAR-10 or CIFAR-100 dataset or create a synthetic dataset if download fails.
    
    Args:
        dataset_name: 'cifar10' or 'cifar100'
        batch_size: Batch size for data loaders
        use_synthetic: Whether to use synthetic data
    """
    # Define transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Class names and number of classes
    if dataset_name.lower() == 'cifar10':
        num_classes = 10
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        dataset_class = torchvision.datasets.CIFAR10
    elif dataset_name.lower() == 'cifar100':
        num_classes = 100
        # CIFAR-100 has 100 classes grouped into 20 superclasses
        classes = None  # Too many to list here
        dataset_class = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'cifar10' or 'cifar100'.")
    
    if not use_synthetic:
        try:
            # Try to load the real dataset
            print(f"Attempting to download {dataset_name.upper()} dataset...")
            
            # Create a temporary directory for download
            os.makedirs('./data', exist_ok=True)
            
            # Try to download with a timeout
            trainset = dataset_class(
                root='./data', train=True, download=True, transform=transform_train
            )
            
            testset = dataset_class(
                root='./data', train=False, download=True, transform=transform_test
            )
            
            # Create data loaders
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True, num_workers=2
            )
            
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False, num_workers=2
            )
            
            print(f"Successfully loaded {dataset_name.upper()} dataset.")
            return trainloader, testloader, classes, num_classes
            
        except Exception as e:
            print(f"Error loading {dataset_name.upper()} dataset: {e}")
            print("Falling back to synthetic dataset...")
            use_synthetic = True
    
    if use_synthetic:
        print(f"Creating synthetic dataset with structured patterns for {num_classes} classes...")
        # Create synthetic dataset with structured patterns
        # Each class will have a distinct pattern to make learning easier
        
        # Number of samples
        train_samples = 5000  # Reduced for faster training
        test_samples = 1000   # Reduced for faster evaluation
        
        # Image dimensions
        img_size = 32
        channels = 3
        
        # Create structured patterns for each class
        class_patterns = []
        for i in range(num_classes):
            # Create a base pattern for each class
            pattern = torch.zeros(channels, img_size, img_size)
            
            # Generate different patterns based on class index
            pattern_type = i % 10
            
            if pattern_type == 0:  # Horizontal stripes
                for j in range(0, img_size, 4):
                    pattern[:, j:j+2, :] = 1.0
            elif pattern_type == 1:  # Vertical stripes
                for j in range(0, img_size, 4):
                    pattern[:, :, j:j+2] = 1.0
            elif pattern_type == 2:  # Diagonal pattern
                for j in range(img_size):
                    if 0 <= j < img_size and 0 <= j < img_size:
                        pattern[:, j, j] = 1.0
                        if j+1 < img_size:
                            pattern[:, j, j+1] = 1.0
            elif pattern_type == 3:  # Circle pattern
                center = img_size // 2
                for x in range(img_size):
                    for y in range(img_size):
                        if ((x - center) ** 2 + (y - center) ** 2) <= (img_size // 4) ** 2:
                            pattern[:, x, y] = 1.0
            elif pattern_type == 4:  # Cross pattern
                center = img_size // 2
                width = 3
                pattern[:, center-width:center+width, :] = 1.0
                pattern[:, :, center-width:center+width] = 1.0
            elif pattern_type == 5:  # Square pattern
                margin = img_size // 4
                pattern[:, margin:img_size-margin, margin:img_size-margin] = 1.0
            elif pattern_type == 6:  # Checkerboard pattern
                for x in range(0, img_size, 4):
                    for y in range(0, img_size, 4):
                        pattern[:, x:x+2, y:y+2] = 1.0
            elif pattern_type == 7:  # Corner pattern
                corner_size = img_size // 3
                pattern[:, :corner_size, :corner_size] = 1.0
                pattern[:, -corner_size:, -corner_size:] = 1.0
            elif pattern_type == 8:  # Border pattern
                border = 3
                pattern[:, :border, :] = 1.0
                pattern[:, -border:, :] = 1.0
                pattern[:, :, :border] = 1.0
                pattern[:, :, -border:] = 1.0
            else:  # Radial pattern
                center = img_size // 2
                for x in range(img_size):
                    for y in range(img_size):
                        angle = torch.atan2(torch.tensor(y - center), torch.tensor(x - center))
                        if angle.abs() < 0.3 or (torch.pi - angle.abs()) < 0.3:
                            pattern[:, x, y] = 1.0
            
            # Add different color tint for each class
            color_pattern = pattern.clone()
            color_idx = i % 6
            if color_idx == 0:
                color_pattern[0] *= 0.8  # More red
                color_pattern[1] *= 0.2
                color_pattern[2] *= 0.2
            elif color_idx == 1:
                color_pattern[0] *= 0.2
                color_pattern[1] *= 0.8  # More green
                color_pattern[2] *= 0.2
            elif color_idx == 2:
                color_pattern[0] *= 0.2
                color_pattern[1] *= 0.2
                color_pattern[2] *= 0.8  # More blue
            elif color_idx == 3:
                color_pattern[0] *= 0.8  # Purple
                color_pattern[1] *= 0.2
                color_pattern[2] *= 0.8
            elif color_idx == 4:
                color_pattern[0] *= 0.8  # Yellow
                color_pattern[1] *= 0.8
                color_pattern[2] *= 0.2
            else:
                color_pattern[0] *= 0.2  # Cyan
                color_pattern[1] *= 0.8
                color_pattern[2] *= 0.8
                
            class_patterns.append(color_pattern)
        
        # Create training data
        train_data = []
        train_labels = []
        
        # Determine samples per class
        samples_per_class = max(1, train_samples // num_classes)
        
        for i in range(num_classes):
            # Base pattern for this class
            base_pattern = class_patterns[i]
            
            for _ in range(samples_per_class):
                # Add random noise to the pattern
                noise = torch.randn_like(base_pattern) * 0.2
                sample = base_pattern + noise
                
                # Clip values to [0, 1]
                sample = torch.clamp(sample, 0, 1)
                
                # Normalize to [-1, 1]
                sample = (sample * 2) - 1
                
                train_data.append(sample)
                train_labels.append(i)
        
        # Create test data
        test_data = []
        test_labels = []
        
        # Determine samples per class
        samples_per_class = max(1, test_samples // num_classes)
        
        for i in range(num_classes):
            # Base pattern for this class
            base_pattern = class_patterns[i]
            
            for _ in range(samples_per_class):
                # Add random noise to the pattern (more noise for test set)
                noise = torch.randn_like(base_pattern) * 0.3
                sample = base_pattern + noise
                
                # Clip values to [0, 1]
                sample = torch.clamp(sample, 0, 1)
                
                # Normalize to [-1, 1]
                sample = (sample * 2) - 1
                
                test_data.append(sample)
                test_labels.append(i)
        
        # Convert to tensors
        train_data = torch.stack(train_data)
        train_labels = torch.tensor(train_labels)
        test_data = torch.stack(test_data)
        test_labels = torch.tensor(test_labels)
        
        # Create TensorDatasets
        trainset = torch.utils.data.TensorDataset(train_data, train_labels)
        testset = torch.utils.data.TensorDataset(test_data, test_labels)
        
        # Create data loaders
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True
        )
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False
        )
        
        actual_train_samples = len(train_data)
        actual_test_samples = len(test_data)
        print(f"Created synthetic dataset with {actual_train_samples} training samples and {actual_test_samples} test samples.")
        return trainloader, testloader, classes, num_classes

def train_model(model, trainloader, testloader, num_epochs=10, learning_rate=0.001, device='cpu'):
    """
    Train the model on CIFAR.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Add weight decay (L2 regularization) to optimizer
    weight_decay = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True
    )
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    training_times = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Reset model state
            model.reset_state()
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_time = time.time() - start_time
        training_times.append(epoch_time)
        
        # Calculate training statistics
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Evaluate on test set
        test_loss, test_acc = evaluate_model(model, testloader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # Update learning rate based on test loss
        scheduler.step(test_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
              f"Time: {epoch_time:.2f}s")
    
    # Calculate average training time per epoch
    avg_training_time = sum(training_times) / len(training_times)
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'avg_training_time': avg_training_time
    }

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate the model on a dataset.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Reset model state
            model.reset_state()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate statistics
    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def plot_results(results_dict, save_dir='results'):
    """
    Plot training and test results.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(12, 6))
    for model_name, results in results_dict.items():
        plt.plot(results['train_losses'], label=f"{model_name} - Train")
        plt.plot(results['test_losses'], label=f"{model_name} - Test")
    
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_comparison.png'))
    plt.close()
    
    # Plot accuracy
    plt.figure(figsize=(12, 6))
    for model_name, results in results_dict.items():
        plt.plot(results['train_accuracies'], label=f"{model_name} - Train")
        plt.plot(results['test_accuracies'], label=f"{model_name} - Test")
    
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'))
    plt.close()
    
    # Plot training time
    plt.figure(figsize=(10, 6))
    model_names = list(results_dict.keys())
    training_times = [results['avg_training_time'] for results in results_dict.values()]
    
    plt.bar(model_names, training_times)
    plt.title('Average Training Time per Epoch')
    plt.xlabel('Model')
    plt.ylabel('Time (seconds)')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(save_dir, 'training_time_comparison.png'))
    plt.close()
    
    # Plot final test accuracy
    plt.figure(figsize=(10, 6))
    final_accuracies = [results['test_accuracies'][-1] for results in results_dict.values()]
    
    plt.bar(model_names, final_accuracies)
    plt.title('Final Test Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(save_dir, 'final_accuracy_comparison.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Benchmark CIFAR classification')
    parser.add_argument('--dataset_name', type=str, default='cifar10', 
                        choices=['cifar10', 'cifar100'],
                        help='Dataset to use')
    parser.add_argument('--model_type', type=str, default='all', 
                        choices=['lnn', 'dynamic_weight_lnn', 'digital_neocortex', 'all'],
                        help='Model type to benchmark')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--use_synthetic', action='store_true', help='Use synthetic dataset instead of CIFAR')
    
    # LNN specific parameters
    parser.add_argument('--dt', type=float, default=0.1, help='Time step for ODE integration')
    
    # Digital Neocortex specific parameters
    parser.add_argument('--num_experts', type=int, default=4, help='Number of experts in Digital Neocortex')
    parser.add_argument('--sparsity', type=float, default=0.3, help='Sparsity factor in Digital Neocortex')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load CIFAR data or synthetic data
    print("Loading dataset...")
    trainloader, testloader, classes, num_classes = load_cifar_data(
        dataset_name=args.dataset_name, batch_size=args.batch_size, use_synthetic=args.use_synthetic
    )
    
    # Define model types to benchmark
    if args.model_type == 'all':
        model_types = ['lnn', 'dynamic_weight_lnn', 'digital_neocortex']
    else:
        model_types = [args.model_type]
    
    # Results dictionary
    results_dict = {}
    
    # Benchmark each model type
    for model_type in model_types:
        print(f"\n===== Benchmarking {model_type} =====")
        
        # Create model
        if model_type == 'lnn':
            model = CIFARModel(
                model_type=model_type,
                hidden_dim=args.hidden_dim,
                output_dim=num_classes,
                dt=args.dt
            )
        elif model_type == 'dynamic_weight_lnn':
            model = CIFARModel(
                model_type=model_type,
                hidden_dim=args.hidden_dim,
                output_dim=num_classes,
                dt=args.dt
            )
        elif model_type == 'digital_neocortex':
            model = CIFARModel(
                model_type=model_type,
                hidden_dim=args.hidden_dim,
                output_dim=num_classes,
                dt=args.dt,
                num_experts=args.num_experts,
                sparsity=args.sparsity
            )
        
        # Train model
        results = train_model(
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            device=args.device
        )
        
        # Save results
        results_dict[model_type] = results
        
        # Print final results
        print(f"\nFinal results for {model_type}:")
        print(f"  Train Loss: {results['train_losses'][-1]:.4f}")
        print(f"  Test Loss: {results['test_losses'][-1]:.4f}")
        print(f"  Train Accuracy: {results['train_accuracies'][-1]:.2f}%")
        print(f"  Test Accuracy: {results['test_accuracies'][-1]:.2f}%")
        print(f"  Average Training Time: {results['avg_training_time']:.2f}s per epoch")
    
    # Plot results
    print("\nPlotting results...")
    plot_results(results_dict, save_dir=args.save_dir)
    
    print(f"\nResults saved to {args.save_dir}")

if __name__ == "__main__":
    main()
