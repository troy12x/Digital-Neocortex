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

"""
Sequence Memory Task:
This benchmark tests a model's ability to remember information from earlier in a sequence
and use it later to make predictions. It's similar to the copy memory task but with text.

Task Types:
1. Recall: Remember a token from a specific position and reproduce it later
2. Delayed XOR: Perform XOR operation on tokens separated by a delay
3. Associative Recall: Remember key-value pairs and recall values when given keys
"""

# Character vocabulary for encoding
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"

class SequenceMemoryDataset:
    def __init__(self, batch_size=8, seq_len=50, task_type='recall', split='train'):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.task_type = task_type
        self.split = split
        
        # Create character vocabulary
        self.char_to_idx = {char: i+1 for i, char in enumerate(CHARS)}
        self.char_to_idx['<pad>'] = 0
        self.idx_to_char = {i+1: char for i, char in enumerate(CHARS)}
        self.idx_to_char[0] = '<pad>'
        self.vocab_size = len(self.char_to_idx)
        
        # Set random seed for reproducibility
        np.random.seed(42 if split == 'train' else 43)
    
    def __len__(self):
        return 100 if self.split == 'train' else 20
    
    def generate_batch(self):
        """Generate a batch of sequences for the specified task"""
        if self.task_type == 'recall':
            return self._generate_recall_batch()
        elif self.task_type == 'delayed_xor':
            return self._generate_delayed_xor_batch()
        elif self.task_type == 'associative_recall':
            return self._generate_associative_recall_batch()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def _generate_recall_batch(self):
        """
        Generate a batch for the recall task.
        
        Format:
        Input: "recall X in Y steps" followed by a sequence of characters
        Target: Predict the character at position X after Y steps
        
        Example:
        Input: "recall 3 in 10 steps abcdefghij"
        Target: At the end, predict 'c' (the character at position 3)
        """
        batch_inputs = []
        batch_targets = []
        
        for _ in range(self.batch_size):
            # Choose a random position to recall (1-indexed for human readability)
            recall_pos = np.random.randint(1, 10)
            # Choose a random delay (steps before we need to recall)
            delay = np.random.randint(15, 30)
            
            # Create the instruction part
            instruction = f"recall {recall_pos} in {delay} steps "
            
            # Create a random sequence of characters
            seq_length = self.seq_len - len(instruction) - delay
            if seq_length < 10:  # Ensure we have enough characters
                seq_length = 10
            
            char_seq = ''.join(np.random.choice(list(CHARS), seq_length))
            
            # The character to recall (0-indexed in the implementation)
            target_char = char_seq[recall_pos-1]
            
            # Create the full input sequence
            input_seq = instruction + char_seq + '<pad>' * delay
            
            # Truncate or pad to match seq_len
            if len(input_seq) < self.seq_len:
                input_seq = input_seq + '<pad>' * (self.seq_len - len(input_seq))
            else:
                input_seq = input_seq[:self.seq_len]
            
            batch_inputs.append(input_seq)
            batch_targets.append(target_char)
        
        # Convert to one-hot encoding
        batch_one_hot = np.zeros((self.batch_size, self.seq_len, self.vocab_size))
        for i, seq in enumerate(batch_inputs):
            for t, char in enumerate(seq):
                if t < self.seq_len:
                    char_idx = self.char_to_idx.get(char, 0)
                    batch_one_hot[i, t, char_idx] = 1.0
        
        # Convert targets to indices
        target_indices = np.array([self.char_to_idx.get(char, 0) for char in batch_targets])
        
        # Convert to tensors
        inputs = torch.FloatTensor(batch_one_hot)
        targets = torch.LongTensor(target_indices)
        
        return inputs, targets
    
    def _generate_delayed_xor_batch(self):
        """
        Generate a batch for the delayed XOR task.
        
        Format:
        Input: "xor at 5 and 15" followed by a sequence of binary digits
        Target: Predict the XOR of the digits at positions 5 and 15
        
        Example:
        Input: "xor at 5 and 15 10110101..."
        Target: XOR of digits at positions 5 and 15
        """
        batch_inputs = []
        batch_targets = []
        
        for _ in range(self.batch_size):
            # Choose two random positions for XOR operation
            pos1 = np.random.randint(5, 10)
            pos2 = np.random.randint(15, 25)
            
            # Create the instruction part
            instruction = f"xor at {pos1} and {pos2} "
            
            # Create a random sequence of binary digits
            seq_length = self.seq_len - len(instruction)
            if seq_length < pos2 + 5:  # Ensure we have enough characters
                seq_length = pos2 + 5
            
            bin_seq = ''.join(np.random.choice(['0', '1'], seq_length))
            
            # Calculate the XOR result
            bit1 = int(bin_seq[pos1-1])
            bit2 = int(bin_seq[pos2-1])
            xor_result = str(bit1 ^ bit2)
            
            # Create the full input sequence
            input_seq = instruction + bin_seq
            
            # Truncate or pad to match seq_len
            if len(input_seq) < self.seq_len:
                input_seq = input_seq + '<pad>' * (self.seq_len - len(input_seq))
            else:
                input_seq = input_seq[:self.seq_len]
            
            batch_inputs.append(input_seq)
            batch_targets.append(xor_result)
        
        # Convert to one-hot encoding
        batch_one_hot = np.zeros((self.batch_size, self.seq_len, self.vocab_size))
        for i, seq in enumerate(batch_inputs):
            for t, char in enumerate(seq):
                if t < self.seq_len:
                    char_idx = self.char_to_idx.get(char, 0)
                    batch_one_hot[i, t, char_idx] = 1.0
        
        # Convert targets to indices
        target_indices = np.array([self.char_to_idx.get(char, 0) for char in batch_targets])
        
        # Convert to tensors
        inputs = torch.FloatTensor(batch_one_hot)
        targets = torch.LongTensor(target_indices)
        
        return inputs, targets
    
    def _generate_associative_recall_batch(self):
        """
        Generate a batch for the associative recall task.
        
        Format:
        Input: Several key-value pairs followed by a query key
        Target: Predict the value associated with the query key
        
        Example:
        Input: "a:1 b:2 c:3 ... query:b"
        Target: Predict '2' (the value associated with key 'b')
        """
        batch_inputs = []
        batch_targets = []
        
        for _ in range(self.batch_size):
            # Create random key-value pairs
            num_pairs = np.random.randint(5, 10)
            pairs = {}
            
            # Generate unique keys
            keys = np.random.choice(list(CHARS[:26]), num_pairs, replace=False)
            values = np.random.choice(list(CHARS[26:]), num_pairs, replace=False)
            
            for i in range(num_pairs):
                pairs[keys[i]] = values[i]
            
            # Create the input sequence with key-value pairs
            input_seq = ""
            for k, v in pairs.items():
                input_seq += f"{k}:{v} "
            
            # Choose a random key to query
            query_key = np.random.choice(list(pairs.keys()))
            target_value = pairs[query_key]
            
            # Add the query
            input_seq += f"query:{query_key}"
            
            # Pad to match seq_len
            if len(input_seq) < self.seq_len:
                input_seq = input_seq + '<pad>' * (self.seq_len - len(input_seq))
            else:
                input_seq = input_seq[:self.seq_len]
            
            batch_inputs.append(input_seq)
            batch_targets.append(target_value)
        
        # Convert to one-hot encoding
        batch_one_hot = np.zeros((self.batch_size, self.seq_len, self.vocab_size))
        for i, seq in enumerate(batch_inputs):
            for t, char in enumerate(seq):
                if t < self.seq_len:
                    char_idx = self.char_to_idx.get(char, 0)
                    batch_one_hot[i, t, char_idx] = 1.0
        
        # Convert targets to indices
        target_indices = np.array([self.char_to_idx.get(char, 0) for char in batch_targets])
        
        # Convert to tensors
        inputs = torch.FloatTensor(batch_one_hot)
        targets = torch.LongTensor(target_indices)
        
        return inputs, targets

# Model for sequence memory tasks
class SequenceMemoryModel(nn.Module):
    def __init__(self, base_model, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Process sequence with base model
        base_output = self.base_model(x)
        
        # Handle different return types
        if isinstance(base_output, tuple):
            outputs = base_output[0]  # Extract outputs from tuple
        else:
            outputs = base_output
        
        # Use the final hidden state for classification
        final_hidden = outputs[:, -1, :]
        
        # Classify
        logits = self.classifier(final_hidden)
        
        return logits

def train_model(model, train_dataset, test_dataset, optimizer, num_epochs=5, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
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
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_test_loss = test_loss / len(test_dataset)
        test_losses.append(avg_test_loss)
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return train_losses, test_losses, accuracy

def run_benchmark(task_type='recall', batch_size=8, seq_len=50, hidden_dim=64, num_epochs=5, device='cpu'):
    # Create datasets
    train_dataset = SequenceMemoryDataset(batch_size=batch_size, seq_len=seq_len, task_type=task_type, split='train')
    test_dataset = SequenceMemoryDataset(batch_size=batch_size, seq_len=seq_len, task_type=task_type, split='test')
    
    vocab_size = train_dataset.vocab_size
    print(f"Task: {task_type}, Vocabulary size: {vocab_size}")
    
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
    
    # Wrap with sequence memory model
    base_model = SequenceMemoryModel(base_lnn, vocab_size, hidden_dim, vocab_size)
    dynamic_model = SequenceMemoryModel(dynamic_lnn, vocab_size, hidden_dim, vocab_size)
    neocortex_model = SequenceMemoryModel(digital_neocortex, vocab_size, hidden_dim, vocab_size)
    
    # Train and evaluate models
    results = {}
    
    print(f"Training Base LNN on {task_type} task...")
    optimizer = optim.Adam(base_model.parameters(), lr=0.001)
    train_losses, test_losses, accuracy = train_model(
        base_model, train_dataset, test_dataset, optimizer, num_epochs, device
    )
    results['Base LNN'] = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'accuracy': accuracy
    }
    
    print(f"\nTraining DynamicWeightLNN on {task_type} task...")
    optimizer = optim.Adam(dynamic_model.parameters(), lr=0.001)
    train_losses, test_losses, accuracy = train_model(
        dynamic_model, train_dataset, test_dataset, optimizer, num_epochs, device
    )
    results['DynamicWeightLNN'] = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'accuracy': accuracy
    }
    
    print(f"\nTraining DigitalNeocortex on {task_type} task...")
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
    plot_results(results, task_type, num_epochs)
    
    return results

def plot_results(results, task_type, num_epochs):
    plt.figure(figsize=(15, 10))
    
    # Plot training losses
    plt.subplot(2, 1, 1)
    for model_name, data in results.items():
        plt.plot(range(1, num_epochs+1), data['train_losses'], marker='o', label=f"{model_name} (Train)")
    plt.title(f'{task_type.capitalize()} Task - Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot test losses
    plt.subplot(2, 1, 2)
    for model_name, data in results.items():
        plt.plot(range(1, num_epochs+1), data['test_losses'], marker='o', label=f"{model_name} (Test)")
    plt.title(f'{task_type.capitalize()} Task - Test Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'sequence_memory_{task_type}_results.png')
    plt.close()
    
    # Create bar chart for accuracy
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in model_names]
    
    plt.bar(model_names, accuracies)
    plt.title(f'{task_type.capitalize()} Task - Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 1, f"{acc:.2f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(f'sequence_memory_{task_type}_accuracy.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Benchmark LNN models on sequence memory tasks')
    parser.add_argument('--task', type=str, default='recall', 
                        choices=['recall', 'delayed_xor', 'associative_recall'],
                        help='Type of sequence memory task')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--seq_len', type=int, default=50, help='Sequence length')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_benchmark(
        task_type=args.task,
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
