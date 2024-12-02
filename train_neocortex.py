import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from test_neocortex import TestPatterns

class NeocortexTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.test_patterns = TestPatterns()
        self.neocortex = self.test_patterns.neocortex
        self.baseline_rnn = self.test_patterns.baseline_rnn
        self.baseline_transformer = self.test_patterns.baseline_transformer
        
        # Optimizers
        self.neocortex_optim = optim.Adam(self.neocortex.parameters(), lr=config.learning_rate)
        self.rnn_optim = optim.Adam(self.baseline_rnn.parameters(), lr=config.learning_rate)
        self.transformer_optim = optim.Adam(self.baseline_transformer.parameters(), lr=config.learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, model, optimizer, patterns, epoch):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(patterns.items(), desc=f'Epoch {epoch}')
        for pattern_name, pattern in progress_bar:
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(model, type(self.neocortex)):
                output = model.process_input(pattern, 'test')
            else:
                output = model(pattern)
            
            # Ensure dimensions match
            pattern_flat = pattern.view(pattern.size(0), -1)
            output_flat = output.view(output.size(0), -1)
            
            # Compute loss
            loss = self.criterion(output_flat, pattern_flat)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        return total_loss / len(patterns)
    
    def train_all_models(self, num_epochs=10):
        patterns = self.test_patterns.generate_test_patterns()
        all_patterns = {}
        for category in patterns.values():
            all_patterns.update(category)
        
        results = {
            'neocortex': [],
            'rnn': [],
            'transformer': []
        }
        
        print("\nTraining Models:")
        print("=" * 50)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train Neocortex
            print("\nTraining Neocortex:")
            neocortex_loss = self.train_epoch(
                self.neocortex, 
                self.neocortex_optim, 
                all_patterns, 
                epoch
            )
            results['neocortex'].append(neocortex_loss)
            
            # Train RNN
            print("\nTraining RNN:")
            rnn_loss = self.train_epoch(
                self.baseline_rnn, 
                self.rnn_optim, 
                all_patterns, 
                epoch
            )
            results['rnn'].append(rnn_loss)
            
            # Train Transformer
            print("\nTraining Transformer:")
            transformer_loss = self.train_epoch(
                self.baseline_transformer, 
                self.transformer_optim, 
                all_patterns, 
                epoch
            )
            results['transformer'].append(transformer_loss)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Neocortex Loss: {neocortex_loss:.4f}")
            print(f"RNN Loss: {rnn_loss:.4f}")
            print(f"Transformer Loss: {transformer_loss:.4f}")
            
            # Test models
            if (epoch + 1) % 5 == 0:
                print("\nRunning Tests...")
                self.test_patterns.test_pattern_recognition()
                
        return results

def main():
    from training_utils import TrainingConfig
    
    # Create config
    config = TrainingConfig()
    config.learning_rate = 0.001
    config.num_epochs = 20
    
    # Initialize trainer
    trainer = NeocortexTrainer(config)
    
    # Train models
    results = trainer.train_all_models(config.num_epochs)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    print("=" * 50)
    trainer.test_patterns.test_pattern_recognition()
    
    print("\nTraining Complete!")

if __name__ == "__main__":
    main() 