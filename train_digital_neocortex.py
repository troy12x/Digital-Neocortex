import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json
from tqdm import tqdm
from datasets import load_dataset
from collections import Counter

from digital_neocortex_components import DigitalNeocortex

class DigitalNeocortexTokenizer:
    """
    A simple tokenizer for the Digital Neocortex model.
    """
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<sos>': 2,
            '<eos>': 3
        }
        
        # Initialize special tokens
        for token, idx in self.special_tokens.items():
            self.word_to_idx[token] = idx
            self.idx_to_word[idx] = token
        
        self.next_idx = len(self.special_tokens)
    
    def fit(self, texts):
        """
        Build vocabulary from a list of texts.
        """
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Select top words by frequency
        vocab_size = min(self.vocab_size, len(word_counts) + len(self.special_tokens))
        most_common = word_counts.most_common(vocab_size - len(self.special_tokens))
        
        # Add to vocabulary
        for word, _ in most_common:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = self.next_idx
                self.idx_to_word[self.next_idx] = word
                self.next_idx += 1
        
        print(f"Vocabulary size: {len(self.word_to_idx)}")
    
    def encode(self, text, max_length=None, padding=True):
        """
        Convert text to token indices.
        """
        words = text.lower().split()
        
        # Add start token
        tokens = [self.special_tokens['<sos>']]
        
        # Convert words to indices
        for word in words:
            if word in self.word_to_idx:
                tokens.append(self.word_to_idx[word])
            else:
                tokens.append(self.special_tokens['<unk>'])
        
        # Add end token
        tokens.append(self.special_tokens['<eos>'])
        
        # Truncate if needed
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Pad if needed
        if padding and max_length is not None and len(tokens) < max_length:
            tokens += [self.special_tokens['<pad>']] * (max_length - len(tokens))
        
        return tokens
    
    def decode(self, tokens):
        """
        Convert token indices back to text.
        """
        words = []
        for token in tokens:
            if token in self.idx_to_word and token != self.special_tokens['<pad>']:
                word = self.idx_to_word[token]
                if word not in ['<sos>', '<eos>', '<pad>', '<unk>']:
                    words.append(word)
        
        return ' '.join(words)
    
    def save(self, path):
        """
        Save tokenizer to file.
        """
        tokenizer_data = {
            'vocab_size': self.vocab_size,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': {int(k): v for k, v in self.idx_to_word.items()},
            'special_tokens': self.special_tokens,
            'next_idx': self.next_idx
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path):
        """
        Load tokenizer from file.
        """
        with open(path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        tokenizer = cls(vocab_size=tokenizer_data['vocab_size'])
        tokenizer.word_to_idx = tokenizer_data['word_to_idx']
        tokenizer.idx_to_word = {int(k): v for k, v in tokenizer_data['idx_to_word'].items()}
        tokenizer.special_tokens = tokenizer_data['special_tokens']
        tokenizer.next_idx = tokenizer_data['next_idx']
        
        return tokenizer

class DigitalNeocortexModel(nn.Module):
    """
    Digital Neocortex model with embedding layer for text processing.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_experts=4, sparsity=0.3, dt=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.digital_neocortex = DigitalNeocortex(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_experts=num_experts,
            sparsity=sparsity,
            dt=dt
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        # Embed input
        embedded = self.embedding(x)
        
        # Process through Digital Neocortex
        hidden = self.digital_neocortex(embedded)
        
        # Project to output dimension
        output = self.output_layer(hidden)
        
        return output
    
    def reset_state(self):
        """Reset internal states"""
        self.digital_neocortex.reset_state()
    
    def save(self, path):
        """
        Save model to file.
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.embedding.num_embeddings,
            'embedding_dim': self.embedding.embedding_dim,
            'hidden_dim': self.digital_neocortex.hidden_dim,
            'output_dim': self.output_layer.out_features,
            'num_experts': self.digital_neocortex.num_experts,
            'sparsity': self.digital_neocortex.sparsity,
            'dt': self.digital_neocortex.dt
        }, path)
    
    @classmethod
    def load(cls, path, device='cpu'):
        """
        Load model from file.
        """
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            output_dim=checkpoint['output_dim'],
            num_experts=checkpoint['num_experts'],
            sparsity=checkpoint['sparsity'],
            dt=checkpoint['dt']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

def prepare_dataset(tokenizer, max_seq_len=50):
    """
    Prepare the dataset for training.
    """
    # Load dataset from Hugging Face
    print("Loading dataset...")
    dataset = load_dataset("miscovery/arabic_egypt_english_world_facts")
    
    # Check available splits
    print(f"Available splits: {dataset.keys()}")
    
    # Extract train data
    train_dataset = dataset['train']
    
    # Create train/test split (80/20)
    train_test_split = train_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Extract English questions
    train_texts = train_dataset['en_question']
    test_texts = test_dataset['en_question']
    
    # Fit tokenizer on training data
    tokenizer.fit(train_texts)
    
    # Tokenize datasets
    train_encodings = [tokenizer.encode(text, max_length=max_seq_len) for text in train_texts]
    test_encodings = [tokenizer.encode(text, max_length=max_seq_len) for text in test_texts]
    
    return train_encodings, test_encodings, tokenizer

class TextDataset:
    """
    Dataset for next token prediction.
    """
    def __init__(self, encodings, batch_size=16, context_length=10):
        self.encodings = encodings
        self.batch_size = batch_size
        self.context_length = context_length
    
    def __len__(self):
        return (len(self.encodings) + self.batch_size - 1) // self.batch_size
    
    def generate_batch(self):
        """
        Generate a batch for next token prediction.
        """
        # Randomly select sequences
        indices = np.random.choice(len(self.encodings), self.batch_size, replace=True)
        batch_encodings = [self.encodings[i] for i in indices]
        
        # Create input-target pairs for next token prediction
        batch_inputs = []
        batch_targets = []
        
        for encoding in batch_encodings:
            # Skip sequences that are too short
            if len(encoding) <= self.context_length + 1:
                continue
                
            # Randomly select a position
            max_start = len(encoding) - self.context_length - 1
            if max_start <= 0:
                continue
                
            start_pos = np.random.randint(0, max_start)
            
            # Input: tokens from start_pos to start_pos + context_length
            input_seq = encoding[start_pos:start_pos + self.context_length]
            # Target: token at start_pos + context_length
            target = encoding[start_pos + self.context_length]
            
            batch_inputs.append(input_seq)
            batch_targets.append(target)
        
        # Convert to tensors
        if not batch_inputs:  # Handle empty batch
            return torch.zeros(0, self.context_length), torch.zeros(0)
            
        inputs = torch.LongTensor(batch_inputs)
        targets = torch.LongTensor(batch_targets)
        
        return inputs, targets

def train_model(model, train_dataset, test_dataset, tokenizer, num_epochs=10, learning_rate=0.001, device='cpu'):
    """
    Train the Digital Neocortex model.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for _ in tqdm(range(len(train_dataset)), desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Generate batch
            inputs, targets = train_dataset.generate_batch()
            if inputs.size(0) == 0:  # Skip empty batches
                continue
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Reset model state
            model.reset_state()
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions for the last token in each sequence
            logits = outputs[:, -1, :]
            
            # Compute loss
            loss = criterion(logits, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
            for _ in range(min(len(test_dataset), 50)):  # Limit test evaluation for speed
                # Generate batch
                inputs, targets = test_dataset.generate_batch()
                if inputs.size(0) == 0:  # Skip empty batches
                    continue
                    
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Reset model state
                model.reset_state()
                
                # Forward pass
                outputs = model(inputs)
                
                # Get predictions for the last token in each sequence
                logits = outputs[:, -1, :]
                
                # Compute loss
                loss = criterion(logits, targets)
                test_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_test_loss = test_loss / min(len(test_dataset), 50)
        test_losses.append(avg_test_loss)
        accuracy = 100 * correct / total if total > 0 else 0
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Generate some examples
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            generate_examples(model, tokenizer, device)
    
    return train_losses, test_losses

def generate_examples(model, tokenizer, device, num_examples=3, max_length=50):
    """
    Generate example predictions from the model.
    """
    model.eval()
    
    print("\nGenerated Examples:")
    for i in range(num_examples):
        # Start with a random prompt from special tokens
        prompt = "<sos> what"
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, max_length=10)
        input_tensor = torch.LongTensor([input_ids]).to(device)
        
        # Reset model state
        model.reset_state()
        
        # Generate tokens
        generated = input_ids.copy()
        
        for _ in range(max_length - len(input_ids)):
            # Forward pass
            with torch.no_grad():
                outputs = model(input_tensor)
            
            # Get next token prediction
            next_token_logits = outputs[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            
            # Stop if end of sequence
            if next_token == tokenizer.special_tokens['<eos>']:
                break
            
            # Add to generated sequence
            generated.append(next_token)
            
            # Update input for next iteration
            input_tensor = torch.LongTensor([generated[-10:]]).to(device)
        
        # Decode generated sequence
        generated_text = tokenizer.decode(generated)
        print(f"Example {i+1}: {generated_text}")

def plot_training_results(train_losses, test_losses, save_path='training_results.png'):
    """
    Plot training and test losses.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train Digital Neocortex on Hugging Face dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--context_length', type=int, default=10, help='Context length for next token prediction')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_experts', type=int, default=4, help='Number of experts in Digital Neocortex')
    parser.add_argument('--sparsity', type=float, default=0.3, help='Sparsity factor in Digital Neocortex')
    parser.add_argument('--dt', type=float, default=0.05, help='Time step for ODE integration')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--save_dir', type=str, default='pretrained', help='Directory to save model and tokenizer')
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = DigitalNeocortexTokenizer(vocab_size=10000)
    
    # Prepare dataset
    print("Preparing dataset...")
    train_encodings, test_encodings, tokenizer = prepare_dataset(tokenizer, max_seq_len=50)
    
    # Create datasets
    train_dataset = TextDataset(train_encodings, batch_size=args.batch_size, context_length=args.context_length)
    test_dataset = TextDataset(test_encodings, batch_size=args.batch_size, context_length=args.context_length)
    
    # Initialize model
    model = DigitalNeocortexModel(
        vocab_size=len(tokenizer.word_to_idx),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=len(tokenizer.word_to_idx),
        num_experts=args.num_experts,
        sparsity=args.sparsity,
        dt=args.dt
    )
    
    # Train model
    print("Training model...")
    train_losses, test_losses = train_model(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # Save model and tokenizer
    model_path = os.path.join(args.save_dir, 'digital_neocortex_model.pt')
    tokenizer_path = os.path.join(args.save_dir, 'digital_neocortex_tokenizer.json')
    
    print(f"Saving model to {model_path}")
    model.save(model_path)
    
    print(f"Saving tokenizer to {tokenizer_path}")
    tokenizer.save(tokenizer_path)
    
    # Plot training results
    plot_path = os.path.join(args.save_dir, 'training_results.png')
    plot_training_results(train_losses, test_losses, save_path=plot_path)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
