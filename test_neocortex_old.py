import torch
import torch.nn as nn
import numpy as np
from app import DigitalNeocortex
from training_utils import TrainingConfig

class BaselineRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True
        )
        self.output = nn.Linear(config.hidden_dim, 1)
    
    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(-1)
        output, _ = self.rnn(x)
        return self.output(output)

class BaselineTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Linear(1, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        self.output = nn.Linear(config.hidden_dim, 1)
    
    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output(x)

class TestPatterns:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize configuration
        self.config = TrainingConfig(
            hidden_dim=64,
            num_layers=4,
            num_heads=8,
            dropout=0.1,
            batch_size=32,
            learning_rate=0.001
        )
        
        # Initialize models
        self.neocortex = DigitalNeocortex(self.config).to(self.device)
        self.baseline_rnn = nn.LSTM(64, 64, num_layers=2, batch_first=True).to(self.device)
        self.baseline_transformer = nn.Transformer(
            d_model=64,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256
        ).to(self.device)
        
    def generate_test_patterns(self):
        """Generate all test patterns"""
        return {
            'chirp': self.generate_chirp(),
            'burst': self.generate_burst(),
            'am_signal': self.generate_am_signal()
        }
        
    def generate_chirp(self):
        """Generate a chirp signal"""
        t = torch.linspace(0, 1, 32)
        f0, f1 = 1, 10  # Start and end frequencies
        phase = 2 * np.pi * t * (f0 + (f1-f0) * t / 2)
        chirp = torch.sin(phase)
        
        # Reshape to match expected dimensions [batch, height, width, channels]
        chirp = chirp.view(1, 32, 1, 1)
        chirp = chirp.repeat(1, 1, 32, 64)
        return chirp.to(self.device)
    
    def generate_burst(self):
        """Generate a burst signal"""
        t = torch.linspace(0, 1, 32)
        burst = torch.exp(-(t - 0.5)**2 / 0.01) * torch.sin(2 * np.pi * 10 * t)
        
        # Reshape to match expected dimensions [batch, height, width, channels]
        burst = burst.view(1, 32, 1, 1)
        burst = burst.repeat(1, 1, 32, 64)
        return burst.to(self.device)
    
    def generate_am_signal(self):
        """Generate an amplitude modulated signal"""
        t = torch.linspace(0, 1, 32)
        carrier = torch.sin(2 * np.pi * 10 * t)
        modulator = 0.5 * (1 + torch.sin(2 * np.pi * 2 * t))
        am = modulator * carrier
        
        # Reshape to match expected dimensions [batch, height, width, channels]
        am = am.view(1, 32, 1, 1)
        am = am.repeat(1, 1, 32, 64)
        return am.to(self.device)

def run_comparison_tests():
    tester = TestPatterns()
    
    # Run pattern recognition tests
    print("\nRunning Pattern Recognition Tests")
    print("=" * 50)
    pattern_results = tester.test_pattern_recognition()
    
    # Run comparison tests
    print("\nRunning Model Comparison Tests")
    print("=" * 50)
    comparison_results = tester.compare_models()
    
    # Print summary
    print("\nTEST SUMMARY")
    print("=" * 50)
    
    print("\nPattern Recognition Results:")
    for category, results in pattern_results.items():
        print(f"\n{category.upper()} Patterns:")
        for pattern_name, metrics in results.items():
            print(f"  {pattern_name}:")
            print(f"    Error: {metrics['reconstruction_error']:.4f}")
            print(f"    Similarity: {metrics['pattern_similarity']:.4f}")
    
    print("\nModel Comparison Results:")
    for category, results in comparison_results.items():
        print(f"\n{category.upper()} Patterns:")
        for pattern_name, metrics in results.items():
            print(f"  {pattern_name}:")
            print(f"    Neocortex: {metrics['neocortex']:.4f}")
            print(f"    RNN: {metrics['rnn']:.4f}")
            print(f"    Transformer: {metrics['transformer']:.4f}")

if __name__ == "__main__":
    run_comparison_tests()  