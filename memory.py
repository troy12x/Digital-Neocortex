import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = 4
        self.head_dim = hidden_dim // self.num_heads
        
        # Only value projections - no query/key
        self.v_projs = nn.ModuleList([
            nn.Linear(hidden_dim, self.head_dim) for _ in range(self.num_heads)
        ])
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Fixed attention patterns
        patterns = torch.zeros(4, 32, 32)
        
        # Head 0: Strict local (no learning)
        window = 2
        for i in range(32):
            start = max(0, i-window)
            end = min(32, i+window+1)
            patterns[0, i, start:end] = 1.0
            
        # Head 1: Skip connections
        for i in range(32):
            patterns[1, i, i] = 1.0
            if i + 4 < 32:
                patterns[1, i, i+4] = 1.0
            if i - 4 >= 0:
                patterns[1, i, i-4] = 1.0
                
        # Head 2: Global tokens only
        patterns[2, :, 0] = 1.0  # First token
        patterns[2, :, -1] = 1.0  # Last token
        
        # Head 3: Causal mask
        patterns[3] = torch.tril(torch.ones(32, 32))
        
        # Normalize each row
        patterns = patterns / patterns.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        self.register_buffer('patterns', patterns)
        
        # For metric computation
        self.last_attention_patterns = None
        
    def forward(self, x):
        B, L, D = x.shape
        
        # Process values only
        outputs = []
        attention_patterns = []
        
        for head in range(self.num_heads):
            v = self.v_projs[head](x)
            pattern = self.patterns[head, :L, :L].unsqueeze(0)
            out = torch.matmul(pattern, v)
            outputs.append(out)
            attention_patterns.append(pattern)
        
        # Store patterns for metric computation
        self.last_attention_patterns = torch.stack(attention_patterns, dim=1).squeeze(0)
        
        # Combine heads
        out = torch.cat(outputs, dim=-1)
        out = self.o_proj(out)
        
        return out
    
    def get_attention_stats(self):
        """Return attention patterns for metric computation"""
        if self.last_attention_patterns is None:
            return self.patterns
        return self.last_attention_patterns

class ShortTermMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.memory_size = config.memory_size
        
        # Memory components with careful initialization
        self.memory = nn.Parameter(torch.zeros(config.memory_size, config.hidden_dim))
        nn.init.normal_(self.memory, std=0.01)
        
        # Stable transformations
        self.query_net = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.key_net = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.value_net = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output processing
        self.output_net = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 2),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(config.hidden_dim)
        )
        
        # Memory tracking
        self.register_buffer('usage', torch.zeros(config.memory_size))
        self.register_buffer('memory_buffer', torch.zeros(config.memory_size, config.hidden_dim))
        
        # Add pattern decomposition
        self.pattern_decomposer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim * 3)  # Separate into components
        )
        
        # Pattern-specific processors
        self.am_processor = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.chirp_processor = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.composite_processor = nn.Linear(config.hidden_dim, config.hidden_dim)
        
    def update(self, x):
        # Decompose pattern into components
        components = self.pattern_decomposer(x)
        am_comp, chirp_comp, composite_comp = torch.chunk(components, 3, dim=-1)
        
        # Process each component
        am_processed = self.am_processor(am_comp)
        chirp_processed = self.chirp_processor(chirp_comp)
        composite_processed = self.composite_processor(composite_comp)
        
        # Combine processed components
        processed = am_processed + chirp_processed + composite_processed
        
        # Process queries and memory
        queries = self.query_net(processed)
        keys = self.key_net(self.memory)
        values = self.value_net(self.memory)
        
        # Compute attention with stability measures
        scores = torch.matmul(queries.view(-1, self.hidden_dim), keys.t())
        scores = scores / math.sqrt(self.hidden_dim)
        scores = torch.clamp(scores, min=-5, max=5)
        
        # Get access weights
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=0.1, training=self.training)
        
        # Read from memory
        read = torch.matmul(attn, values)
        batch_size = attn.size(0)
        seq_len = attn.size(1) 
        read = read.view(batch_size, seq_len, -1)
        
        # Update usage
        self.usage.data = 0.9 * self.usage + 0.1 * attn.sum(0).detach()
        
        # Update memory buffer
        self.memory_buffer.copy_(self.memory.data)
        
        # Generate output with residual
        combined = torch.cat([processed, read], dim=-1)
        out = self.output_net(combined)
        
        return out

class LongTermMemory(nn.Module):
    def __init__(self, hidden_dim: int, memory_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        
        # Memory with stable initialization
        self.memory = nn.Parameter(torch.zeros(memory_size, hidden_dim))
        nn.init.normal_(self.memory, std=0.01)
        
        # Stable transformations
        self.query_net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.key_net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.value_net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output processing
        self.output_net = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim)
        )
        
        # Memory tracking
        self.register_buffer('usage', torch.zeros(memory_size))
        self.register_buffer('memory_buffer', torch.zeros(memory_size, hidden_dim))
        self.register_buffer('episodic_memory', torch.zeros(memory_size, hidden_dim))
        
    def forward(self, x):
        B, L, D = x.shape
        
        # Process queries and memory
        queries = self.query_net(x)
        keys = self.key_net(self.memory)
        values = self.value_net(self.memory)
        
        # Compute attention with stability measures
        scores = torch.matmul(queries.view(-1, D), keys.t())
        scores = scores / math.sqrt(D)
        scores = torch.clamp(scores, min=-5, max=5)
        
        # Get access weights
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=0.1, training=self.training)
        
        # Read from memory
        read = torch.matmul(attn, values)
        read = read.view(B, L, D)
        
        # Update usage
        self.usage.data = 0.9 * self.usage + 0.1 * attn.sum(0).detach()
        
        # Update memory buffers
        self.memory_buffer.copy_(self.memory.data)
        self.episodic_memory.copy_(values.detach())
        
        # Generate output with residual
        combined = torch.cat([x, read], dim=-1)
        out = self.output_net(combined)
        
        return out

class ReasoningEngine(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.LayerNorm(self.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
    def forward(self, x):
        return x + 0.1 * self.net(x)  # Scaled residual
