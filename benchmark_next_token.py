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
Next Token Prediction Task:
This benchmark tests a model's ability to learn simple patterns in text
and predict the next token in a sequence.

The model is trained on simple phrases like "what is your name" and then
tested on partial phrases like "what is your" to see if it can predict "name".
"""

# Training phrases
TRAINING_PHRASES = [
    "what is your name",
    "how are you doing",
    "nice to meet you",
    "thank you very much",
    "have a good day",
    "see you tomorrow morning",
    "please help me understand",
    "can you explain this",
    "where are you from",
    "what time is it now"
]

# Test phrases (incomplete versions of training phrases)
TEST_PHRASES = [
    "what is your",  # Expected: "name"
    "how are you",   # Expected: "doing"
    "nice to meet",  # Expected: "you"
    "thank you very", # Expected: "much"
    "have a good",   # Expected: "day"
    "see you tomorrow", # Expected: "morning"
    "please help me", # Expected: "understand"
    "can you explain", # Expected: "this"
    "where are you", # Expected: "from"
    "what time is it" # Expected: "now"
]

# Expected next tokens for each test phrase
EXPECTED_NEXT_TOKENS = [
    "name",
    "doing",
    "you",
    "much",
    "day",
    "morning",
    "understand",
    "this",
    "from",
    "now"
]

class NextTokenDataset:
    def __init__(self, batch_size=8, split='train'):
        self.batch_size = batch_size
        self.split = split
        
        # Create vocabulary from all words in phrases
        all_words = set()
        for phrase in TRAINING_PHRASES + TEST_PHRASES + EXPECTED_NEXT_TOKENS:
            all_words.update(phrase.split())
        
        self.word_to_idx = {word: i+1 for i, word in enumerate(sorted(all_words))}
        self.word_to_idx['<pad>'] = 0
        self.idx_to_word = {i+1: word for i, word in enumerate(sorted(all_words))}
        self.idx_to_word[0] = '<pad>'
        self.vocab_size = len(self.word_to_idx)
        
        # Set random seed for reproducibility
        np.random.seed(42 if split == 'train' else 43)
    
    def __len__(self):
        if self.split == 'train':
            return 100  # Number of training batches
        else:
            return len(TEST_PHRASES)  # Number of test phrases
    
    def generate_batch(self):
        """Generate a batch of sequences for next token prediction"""
        if self.split == 'train':
            return self._generate_train_batch()
        else:
            return self._generate_test_batch()
    
    def _generate_train_batch(self):
        """Generate a batch for training on complete phrases"""
        batch_inputs = []
        batch_targets = []
        
        # Randomly select phrases for this batch
        phrase_indices = np.random.choice(len(TRAINING_PHRASES), self.batch_size, replace=True)
        
        for idx in phrase_indices:
            phrase = TRAINING_PHRASES[idx]
            words = phrase.split()
            
            # For each position in the phrase, predict the next word
            for i in range(len(words) - 1):
                # Input: words up to position i
                input_seq = words[:i+1]
                # Target: word at position i+1
                target_word = words[i+1]
                
                # Pad input sequence
                padded_input = input_seq + ['<pad>'] * (10 - len(input_seq))
                
                batch_inputs.append(padded_input)
                batch_targets.append(target_word)
        
        # Randomly select a subset if we have too many examples
        if len(batch_inputs) > self.batch_size:
            indices = np.random.choice(len(batch_inputs), self.batch_size, replace=False)
            batch_inputs = [batch_inputs[i] for i in indices]
            batch_targets = [batch_targets[i] for i in indices]
        
        # Convert to one-hot encoding
        batch_one_hot = np.zeros((len(batch_inputs), 10, self.vocab_size))
        for i, seq in enumerate(batch_inputs):
            for t, word in enumerate(seq):
                word_idx = self.word_to_idx.get(word, 0)
                batch_one_hot[i, t, word_idx] = 1.0
        
        # Convert targets to indices
        target_indices = np.array([self.word_to_idx.get(word, 0) for word in batch_targets])
        
        # Convert to tensors
        inputs = torch.FloatTensor(batch_one_hot)
        targets = torch.LongTensor(target_indices)
        
        return inputs, targets
    
    def _generate_test_batch(self):
        """Generate a batch for testing on incomplete phrases"""
        batch_inputs = []
        batch_targets = []
        
        # Use all test phrases
        for i in range(len(TEST_PHRASES)):
            phrase = TEST_PHRASES[i]
            expected_next = EXPECTED_NEXT_TOKENS[i]
            
            # Input: all words in the test phrase
            input_seq = phrase.split()
            # Target: the expected next word
            target_word = expected_next
            
            # Pad input sequence
            padded_input = input_seq + ['<pad>'] * (10 - len(input_seq))
            
            batch_inputs.append(padded_input)
            batch_targets.append(target_word)
        
        # Convert to one-hot encoding
        batch_one_hot = np.zeros((len(batch_inputs), 10, self.vocab_size))
        for i, seq in enumerate(batch_inputs):
            for t, word in enumerate(seq):
                word_idx = self.word_to_idx.get(word, 0)
                batch_one_hot[i, t, word_idx] = 1.0
        
        # Convert targets to indices
        target_indices = np.array([self.word_to_idx.get(word, 0) for word in batch_targets])
        
        # Convert to tensors
        inputs = torch.FloatTensor(batch_one_hot)
        targets = torch.LongTensor(target_indices)
        
        return inputs, targets

# Model for next token prediction
class NextTokenModel(nn.Module):
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

def train_model(model, train_dataset, test_dataset, optimizer, num_epochs=100, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for _ in tqdm(range(len(train_dataset)), desc=f"Epoch {epoch+1}/{num_epochs}", disable=epoch % 10 != 0):
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
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Final test with detailed output
    print("\nDetailed test results:")
    model.eval()
    with torch.no_grad():
        inputs, targets = test_dataset.generate_batch()
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        for i in range(len(TEST_PHRASES)):
            input_phrase = TEST_PHRASES[i]
            expected = EXPECTED_NEXT_TOKENS[i]
            predicted_idx = predicted[i].item()
            predicted_word = test_dataset.idx_to_word.get(predicted_idx, "<unknown>")
            
            is_correct = "✓" if predicted_word == expected else "✗"
            print(f"Input: '{input_phrase}', Expected: '{expected}', Predicted: '{predicted_word}' {is_correct}")
    
    return train_losses, test_losses, accuracy

def run_benchmark(batch_size=8, hidden_dim=64, num_epochs=100, device='cpu'):
    # Create datasets
    train_dataset = NextTokenDataset(batch_size=batch_size, split='train')
    test_dataset = NextTokenDataset(batch_size=batch_size, split='test')
    
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
    
    # Wrap with next token model
    base_model = NextTokenModel(base_lnn, vocab_size, hidden_dim, vocab_size)
    dynamic_model = NextTokenModel(dynamic_lnn, vocab_size, hidden_dim, vocab_size)
    neocortex_model = NextTokenModel(digital_neocortex, vocab_size, hidden_dim, vocab_size)
    
    # Train and evaluate models
    results = {}
    
    print("Training Base LNN on next token prediction...")
    optimizer = optim.Adam(base_model.parameters(), lr=0.001)
    train_losses, test_losses, accuracy = train_model(
        base_model, train_dataset, test_dataset, optimizer, num_epochs, device
    )
    results['Base LNN'] = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'accuracy': accuracy
    }
    
    print("\nTraining DynamicWeightLNN on next token prediction...")
    optimizer = optim.Adam(dynamic_model.parameters(), lr=0.001)
    train_losses, test_losses, accuracy = train_model(
        dynamic_model, train_dataset, test_dataset, optimizer, num_epochs, device
    )
    results['DynamicWeightLNN'] = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'accuracy': accuracy
    }
    
    print("\nTraining DigitalNeocortex on next token prediction...")
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
        # Plot every 5 epochs to reduce clutter
        epochs = range(1, num_epochs+1, 5)
        losses = data['train_losses'][::5]
        plt.plot(epochs, losses, marker='o', label=f"{model_name} (Train)")
    plt.title('Next Token Prediction - Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot test losses
    plt.subplot(2, 1, 2)
    for model_name, data in results.items():
        # Plot every 5 epochs to reduce clutter
        epochs = range(1, num_epochs+1, 5)
        losses = data['test_losses'][::5]
        plt.plot(epochs, losses, marker='o', label=f"{model_name} (Test)")
    plt.title('Next Token Prediction - Test Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('next_token_prediction_results.png')
    plt.close()
    
    # Create bar chart for accuracy
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in model_names]
    
    plt.bar(model_names, accuracies)
    plt.title('Next Token Prediction - Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 1, f"{acc:.2f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig('next_token_prediction_accuracy.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Benchmark LNN models on next token prediction')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_benchmark(
        batch_size=args.batch_size,
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
