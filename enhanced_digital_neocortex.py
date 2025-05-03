import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

@dataclass
class NeocortexConfig:
    """Configuration for the Digital Neocortex"""
    vocab_size: int = 30522         # Vocabulary size (default from BERT)
    input_dim: int = 768            # Input dimension
    hidden_dim: int = 256           # Hidden dimension
    num_layers: int = 4             # Number of hierarchical layers
    num_columns: int = 8            # Number of cortical columns
    memory_size: int = 1024         # Size of memory buffer
    stm_size: int = 64             # Short-term memory size
    ltm_size: int = 512            # Long-term memory size
    num_heads: int = 8             # Number of attention heads
    dropout: float = 0.1           # Dropout rate
    max_sequence_length: int = 512  # Maximum sequence length

class CorticalColumn(nn.Module):
    """Implements a cortical column for specialized processing"""
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(
            config.hidden_dim,
            config.hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=False
        )

    def forward(self, x):
        return self.lstm(x)

class MemorySystem(nn.Module):
    """Implements both short-term and long-term memory systems"""
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            batch_first=True
        )
        
        # Initialize memory as 3D tensor [batch_size, memory_size, hidden_dim]
        self.memory_size = config.memory_size
        self.hidden_dim = config.hidden_dim
        self.memory = nn.Parameter(
            torch.randn(1, config.memory_size, config.hidden_dim)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_length, hidden_dim]
        batch_size = x.size(0)
        
        # Expand memory to match batch size
        expanded_memory = self.memory.expand(batch_size, -1, -1)
        
        # Perform attention
        output, _ = self.attention(
            query=x,
            key=expanded_memory,
            value=expanded_memory
        )
        return output

class HierarchicalProcessor(nn.Module):
    """Implements hierarchical processing layers"""
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.num_layers)
        ])

    def forward(self, x):
        outputs = []
        current = x
        
        for layer in self.layers:
            current = layer(current)
            outputs.append(current)
        
        return outputs

class DigitalNeocortex(nn.Module):
    """Main Digital Neocortex architecture"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(config.max_sequence_length, config.hidden_dim)
        )
        
        # Cortical columns
        self.columns = nn.ModuleList([
            CorticalColumn(config) for _ in range(config.num_columns)
        ])
        
        # Memory systems
        self.memory = MemorySystem(config)
        
        # Hierarchical processor
        self.hierarchy = HierarchicalProcessor(config)
        
        # Output generation
        self.output_generator = nn.Sequential(
            nn.Linear(config.hidden_dim * config.num_layers, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.vocab_size)
        )

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape
        
        # Token embedding and positional encoding
        x = self.token_embedding(input_ids)  # [batch_size, seq_length, hidden_dim]
        positions = self.position_embedding[:seq_length]
        x = x + positions
        
        # Process through cortical columns
        column_outputs = []
        for column in self.columns:
            # Reshape for LSTM: [seq_length, batch_size, hidden_dim]
            column_input = x.transpose(0, 1)
            output, _ = column(column_input)
            # Reshape back: [batch_size, seq_length, hidden_dim]
            output = output.transpose(0, 1)
            column_outputs.append(output)
        
        # Combine column outputs
        combined = torch.stack(column_outputs).mean(dim=0)  # [batch_size, seq_length, hidden_dim]
        
        # Process through memory systems
        memory_output = self.memory(combined)  # [batch_size, seq_length, hidden_dim]
        
        # Hierarchical processing
        hierarchy_outputs = self.hierarchy(memory_output)  # List of [batch_size, seq_length, hidden_dim]
        
        # Combine hierarchical outputs along hidden dimension
        final_representation = torch.cat(hierarchy_outputs, dim=-1)  # [batch_size, seq_length, hidden_dim * num_layers]
        
        # Generate output (use only the last sequence position)
        output = self.output_generator(final_representation[:, -1])  # [batch_size, vocab_size]
        
        return output

    def update_memories(self, new_information):
        """Update long-term memories based on new information"""
        with torch.no_grad():
            # Update LTM with new information
            self.memory.ltm.data = 0.99 * self.memory.ltm + 0.01 * new_information.mean(dim=0)