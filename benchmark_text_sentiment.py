import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm

from liquid_neural_network import LiquidNeuralNetwork, DynamicWeightLNN
from digital_neocortex_components import DigitalNeocortex

# Small sentiment analysis dataset
POSITIVE_SAMPLES = [
    "I love this product",
    "This is excellent",
    "Great experience",
    "Very happy with the results",
    "Fantastic service",
    "Highly recommended",
    "Amazing quality",
    "Wonderful time",
    "Exceeded my expectations",
    "Best purchase ever"
]

NEGATIVE_SAMPLES = [
    "Terrible experience",
    "Very disappointed",
    "Poor quality",
    "Would not recommend",
    "Waste of money",
    "Awful service",
    "Not worth it",
    "Completely useless",
    "Frustrated with this",
    "Worst purchase ever"
]

# Character-level encoding for simplicity
class TextDataset:
    def __init__(self, batch_size=8, seq_len=20, split='train'):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.split = split
        
        # Combine samples
        all_samples = [(text, 1) for text in POSITIVE_SAMPLES] + [(text, 0) for text in NEGATIVE_SAMPLES]
        
        # Shuffle and split into train/test
        np.random.seed(42)
        np.random.shuffle(all_samples)
        
        train_size = int(0.8 * len(all_samples))
        if split == 'train':
            self.samples = all_samples[:train_size]
        else:
            self.samples = all_samples[train_size:]
        
        # Create character vocabulary
        all_chars = set()
        for text, _ in all_samples:
            all_chars.update(text.lower())
        
        self.char_to_idx = {char: i+1 for i, char in enumerate(sorted(all_chars))}
        self.char_to_idx['<pad>'] = 0
        self.vocab_size = len(self.char_to_idx)
        
    def __len__(self):
        return (len(self.samples) + self.batch_size - 1) // self.batch_size
    
    def generate_batch(self):
        indices = np.random.choice(len(self.samples), self.batch_size, replace=True)
        batch_texts = [self.samples[i][0] for i in indices]
        batch_labels = [self.samples[i][1] for i in indices]
        
        # Convert to character indices
        batch_inputs = []
        for text in batch_texts:
            text = text.lower()
            # Pad or truncate to seq_len
            if len(text) < self.seq_len:
                text = text + '<pad>' * (self.seq_len - len(text))
            else:
                text = text[:self.seq_len]
            
            # Convert to indices
            char_indices = [self.char_to_idx.get(char, 0) for char in text]
            batch_inputs.append(char_indices)
        
        # Convert to one-hot encoding
        batch_one_hot = np.zeros((self.batch_size, self.seq_len, self.vocab_size))
        for i, seq in enumerate(batch_inputs):
            for t, char_idx in enumerate(seq):
                if t < self.seq_len:  # Ensure we don't exceed sequence length
                    batch_one_hot[i, t, char_idx] = 1.0
        
        # Convert to tensors
        inputs = torch.FloatTensor(batch_one_hot)
        labels = torch.FloatTensor(batch_labels).unsqueeze(1)
        
        return inputs, labels

# Simple model wrapper for sentiment analysis
class SentimentModel(nn.Module):
    def __init__(self, base_model, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Process sequence with base model
        base_output = self.base_model(x)
        
        # Handle different return types
        # LiquidNeuralNetwork and DynamicWeightLNN return (outputs, hidden_states)
        # DigitalNeocortex returns just outputs
        if isinstance(base_output, tuple):
            outputs = base_output[0]  # Extract outputs from tuple
        else:
            outputs = base_output
        
        # Use the final hidden state for classification
        final_hidden = outputs[:, -1, :]
        
        # Classify
        logits = self.classifier(final_hidden)
        probs = self.sigmoid(logits)
        
        return probs

def train_model(model, train_dataset, test_dataset, optimizer, num_epochs=5, device='cpu'):
    model.to(device)
    criterion = nn.BCELoss()
    
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for _ in tqdm(range(len(train_dataset)), desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = train_dataset.generate_batch()
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_dataset)
        train_losses.append(avg_train_loss)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for _ in range(len(test_dataset)):
                inputs, targets = test_dataset.generate_batch()
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_test_loss = test_loss / len(test_dataset)
        test_losses.append(avg_test_loss)
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return train_losses, test_losses, accuracy

def run_benchmark(batch_size=8, seq_len=20, hidden_dim=64, num_epochs=5, device='cpu'):
    # Create datasets
    train_dataset = TextDataset(batch_size=batch_size, seq_len=seq_len, split='train')
    test_dataset = TextDataset(batch_size=batch_size, seq_len=seq_len, split='test')
    
    vocab_size = train_dataset.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Create models
    base_lnn = LiquidNeuralNetwork(
        input_dim=vocab_size,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        dt=0.1
    )
    
    dynamic_lnn = DynamicWeightLNN(
        input_dim=vocab_size,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        dt=0.1
    )
    
    digital_neocortex = DigitalNeocortex(
        input_dim=vocab_size,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        num_experts=4,
        sparsity=0.3,
        dt=0.1
    )
    
    # Wrap with sentiment classifier
    base_model = SentimentModel(base_lnn, vocab_size, hidden_dim)
    dynamic_model = SentimentModel(dynamic_lnn, vocab_size, hidden_dim)
    neocortex_model = SentimentModel(digital_neocortex, vocab_size, hidden_dim)
    
    # Train and evaluate models
    results = {}
    
    print("Training Base LNN on sentiment analysis...")
    optimizer = optim.Adam(base_model.parameters(), lr=0.001)
    train_losses, test_losses, accuracy = train_model(
        base_model, train_dataset, test_dataset, optimizer, num_epochs, device
    )
    results['Base LNN'] = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'accuracy': accuracy
    }
    
    print("\nTraining DynamicWeightLNN on sentiment analysis...")
    optimizer = optim.Adam(dynamic_model.parameters(), lr=0.001)
    train_losses, test_losses, accuracy = train_model(
        dynamic_model, train_dataset, test_dataset, optimizer, num_epochs, device
    )
    results['DynamicWeightLNN'] = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'accuracy': accuracy
    }
    
    print("\nTraining DigitalNeocortex on sentiment analysis...")
    optimizer = optim.Adam(neocortex_model.parameters(), lr=0.001)
    train_losses, test_losses, accuracy = train_model(
        neocortex_model, train_dataset, test_dataset, optimizer, num_epochs, device
    )
    results['DigitalNeocortex'] = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'accuracy': accuracy
    }
    
    # Plot results
    plot_results(results, num_epochs)
    
    return results

def plot_results(results, num_epochs):
    plt.figure(figsize=(15, 10))
    
    # Plot training losses
    plt.subplot(2, 1, 1)
    for model_name, data in results.items():
        plt.plot(range(1, num_epochs+1), data['train_losses'], marker='o', label=f"{model_name} (Train)")
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot test losses
    plt.subplot(2, 1, 2)
    for model_name, data in results.items():
        plt.plot(range(1, num_epochs+1), data['test_losses'], marker='o', label=f"{model_name} (Test)")
    plt.title('Test Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis_results.png')
    plt.close()
    
    # Create bar chart for accuracy
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in model_names]
    
    plt.bar(model_names, accuracies)
    plt.title('Sentiment Analysis Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 1, f"{acc:.2f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis_accuracy.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Benchmark LNN models on sentiment analysis')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length for text')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_benchmark(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_epochs=args.num_epochs,
        device=args.device
    )
    
    # Print final results
    print("\nFinal Results:")
    for model_name, data in results.items():
        print(f"{model_name}: Accuracy = {data['accuracy']:.2f}%")

if __name__ == "__main__":
    main()
