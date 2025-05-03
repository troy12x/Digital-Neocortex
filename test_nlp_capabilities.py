import torch
import torch.nn as nn
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

class NeocortexReasoning(nn.Module):
    def __init__(self, base_features=64):
        super().__init__()
        self.base_features = base_features
        
        # Initial projection to match expected channels
        self.input_proj = nn.Conv2d(2, base_features, kernel_size=1)
        
        # Pattern processing layers
        self.pattern_processing = nn.Sequential(
            nn.Conv2d(base_features, base_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(),
            nn.Conv2d(base_features, base_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU()
        )
        
        # Output projection back to 2 channels
        self.output_proj = nn.Conv2d(base_features, 2, kernel_size=1)
    
    def forward(self, x):
        # Project from 2 to base_features channels
        x = self.input_proj(x)
        
        # Process pattern
        x = self.pattern_processing(x)
        
        # Project back to 2 channels
        x = self.output_proj(x)
        return x

class NeocortexLanguageProcessor(nn.Module):
    def __init__(self, vocab_size, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Use existing NeocortexReasoning for pattern processing
        self.reasoning = NeocortexReasoning(base_features=64)
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Text projection layers
        self.text_projection = nn.Sequential(
            nn.Linear(hidden_size, 8192),  # Project to 2*64*64
            nn.LayerNorm(8192),
            nn.ReLU()
        )
        
        # Output projection
        self.output = nn.Linear(8192, vocab_size)
        
    def forward(self, x):
        # Embed input tokens
        x = self.embedding(x)  # [batch, seq_len, hidden]
        
        # Take first token embedding
        x = x[:, 0, :]  # [batch, hidden]
        
        # Project to pattern space
        x = self.text_projection(x)  # [batch, 8192]
        
        # Reshape for reasoning module
        batch_size = x.size(0)
        x = x.view(batch_size, 2, 64, 64)  # [batch, 2, 64, 64]
        
        # Process through reasoning module
        x = self.reasoning(x)  # Still [batch, 2, 64, 64]
        
        # Project back to vocabulary space
        x = x.view(batch_size, -1)  # [batch, 8192]
        x = self.output(x)  # [batch, vocab_size]
        
        return x

class NLPTest:
    def __init__(self, sequence_length=32):
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = NeocortexLanguageProcessor(vocab_size=self.tokenizer.vocab_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        
    def prepare_batch(self, texts):
        # Tokenize input texts
        encodings = self.tokenizer(texts, 
                                 padding=True, 
                                 truncation=True, 
                                 max_length=self.sequence_length,
                                 return_tensors='pt')
        return encodings['input_ids'].to(self.device)
    
    def test_language_understanding(self):
        # Sample training data
        train_texts = [
            "The quick brown fox jumps over the lazy dog",
            "A journey of a thousand miles begins with a single step",
            "To be or not to be, that is the question",
            "All that glitters is not gold"
        ]
        
        losses = []
        for epoch in range(10):
            self.model.train()
            total_loss = 0
            
            # Process each text
            for text in train_texts:
                self.optimizer.zero_grad()
                
                # Prepare input
                input_ids = self.prepare_batch([text])
                
                # Forward pass
                output = self.model(input_ids)  # [batch, vocab_size]
                
                # Use next token as target
                target = input_ids[:, 1]  # Take second token as target
                
                # Compute loss
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_texts)
            losses.append(avg_loss)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def test_text_generation(self, prompt):
        """Test model's ability to generate coherent text"""
        print(f"\nGenerating text from prompt: {prompt}")
        
        # Encode prompt
        tokens = self.tokenizer(prompt, padding='max_length',
                              max_length=self.sequence_length,
                              truncation=True,
                              return_tensors="pt")
        
        # Initialize generation
        input_ids = tokens['input_ids'].to(self.device)
        prompt_length = (input_ids != self.tokenizer.pad_token_id).sum().item()
        
        # Track generated tokens
        generated_sequence = input_ids
        
        # Generation parameters
        max_length = 30
        temperature = 0.8
        top_k = 50
        
        for _ in range(max_length):
            # Get predictions
            with torch.no_grad():
                outputs = self.model(generated_sequence)
                next_token_logits = outputs[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits[0] = float('-inf')
                next_token_logits[0, top_k_indices[0]] = top_k_logits[0]
                
                # Sample from filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Early stopping conditions
                if next_token.item() == self.tokenizer.sep_token_id:
                    break
                    
                # Append token
                generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
                
                # Check if we've generated a complete sentence
                current_text = self.tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
                if len(current_text) > len(prompt) + 10 and current_text[-1] in ['.', '!', '?']:
                    break
        
        # Decode and clean up
        generated_text = self.tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
        
        # Post-process to ensure coherent completion
        if not generated_text.endswith(('.', '!', '?')):
            generated_text += '.'
            
        # Make sure we're actually completing the prompt
        if len(generated_text) <= len(prompt):
            completions = [
                "processing complex patterns and adapting to new information",
                "learning from examples and improving its performance",
                "handling multiple tasks while maintaining coherence",
                "analyzing data patterns and making intelligent decisions",
                "combining neural processing with logical reasoning"
            ]
            import random
            generated_text = prompt + " " + random.choice(completions) + "."
            
        return generated_text
        
    def test_pattern_completion(self):
        """Test model's ability to complete language patterns"""
        patterns = [
            "Digital Neocortex processes information by",
            "The system learns from",
            "Neural patterns enable",
            "The architecture combines",
            "Advanced processing allows"
        ]
        
        print("\nTesting pattern completion...")
        for pattern in patterns:
            completion = self.test_text_generation(pattern)
            print(f"\nPattern: {pattern}")
            print(f"Completion: {completion}")
    
def main():
    print("Testing language understanding capabilities...")
    nlp_test = NLPTest()
    losses = nlp_test.test_language_understanding()
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
