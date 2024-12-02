import torch
import torch.nn as nn
import math
from typing import Optional
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Only value projections
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Create fixed attention patterns
        patterns = torch.zeros(num_heads + 1, 32, 32)
        
        # Head 0: Local window (diagonal band)
        window = 2
        for i in range(32):
            for j in range(max(0, i-window), min(32, i+window+1)):
                patterns[0, i, j] = 1.0 / (2 * window + 1)
                
        # Head 1: Skip connections (sparse)
        for i in range(32):
            patterns[1, i, i] = 0.5  # Self
            if i + 4 < 32:
                patterns[1, i, i+4] = 0.25  # Forward
            if i - 4 >= 0:
                patterns[1, i, i-4] = 0.25  # Backward
                
        # Head 2: Global tokens
        patterns[2, :, 0] = 0.4  # First token
        patterns[2, :, -1] = 0.4  # Last token
        patterns[2].diagonal().fill_(0.2)  # Self attention
        
        # Head 3: Forward-looking triangular
        for i in range(32):
            end = min(32, i + 4)
            size = end - i
            patterns[3, i, i:end] = 1.0 / size
            
        # Add new frequency-sensitive head
        for i in range(32):
            # Create frequency-based attention window
            freq_window = torch.exp(-0.5 * (torch.arange(32) - i).float() ** 2 / 8)
            patterns[4, i] = freq_window / freq_window.sum()
        
        self.register_buffer('attention_patterns', patterns)
        
        # Add frequency analysis layer
        self.freq_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Analyze frequency components
        freq_weights = torch.sigmoid(self.freq_analyzer(x))
        
        # Project values only
        v = self.v_proj(x).view(batch_size, -1, self.num_heads, self.head_dim)
        v = v.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        
        # Get fixed attention patterns for this sequence length
        patterns = self.attention_patterns[:, :seq_len, :seq_len]
        patterns = patterns.unsqueeze(0)  # [1, heads, seq_len, seq_len]
        
        # Apply attention directly
        output = torch.matmul(patterns, v)  # [batch, heads, seq_len, head_dim]
        
        # Combine heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_dim)
        output = self.o_proj(output)
        
        # Store patterns for metrics
        self.last_attn_weights = patterns.detach()
        
        return output