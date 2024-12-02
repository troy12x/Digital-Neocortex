import numpy as np
import torch
import torch.nn as nn
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import torch.nn.functional as F
from memory import (
    ShortTermMemory, 
    LongTermMemory, 
    ReasoningEngine
)
from learning import (
    LearningManager,
    AdaptationModule,
    ContextAnalyzer,
    OutputGenerator,
    FeedbackProcessor
)
from attention import MultiHeadAttention
from processors import (
    TextProcessor, VisionProcessor, AudioProcessor, SensoryProcessor,
    IntermediateProcessor, AssociativeProcessor
)

@dataclass
class NeocortexConfig:
    """Configuration for the Digital Neocortex"""
    batch_size: int = 32
    input_dim: int = 64
    hidden_dim: int = 64
    num_layers: int = 3
    num_heads: int = 8
    sequence_length: int = 64
    spatial_size: int = 8  # H=W=8 for spatial dimensions
    num_columns: int = 4
    memory_capacity: int = 1000
    dropout: float = 0.1
    learning_rate: float = 1e-4
    action_dim: int = 10  # Default value for action space dimension
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert self.spatial_size in [8, 16, 32], "spatial_size must be 8, 16, or 32"

class BaseProcessor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.input_projection = nn.Linear(dim, dim)
        self.linear = nn.Linear(dim, dim)
        self.vision_projection = nn.Linear(3, dim)  # For vision input
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 4:  # Vision input
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
            x = x.reshape(b, h*w, c)    # [B, H*W, C]
            x = self.vision_projection(x)  # Project channels to hidden dim
            
        x = self.input_projection(x)
        return self.linear(x)

class SensoryProcessor(BaseProcessor): pass
class PatternProcessor(BaseProcessor): pass
class AssociativeProcessor(BaseProcessor): pass
class AbstractProcessor(BaseProcessor): pass
class ExecutiveProcessor(BaseProcessor): pass
class OutputProcessor(BaseProcessor): pass
class TextProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, x):
        # Process text input
        return x

class VisionProcessor(BaseProcessor): pass
class AudioProcessor(BaseProcessor): pass

class TokenEmbedder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

class HierarchicalPositionalEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.position_embedding

class LocalInhibitionCircuit(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.inhibition_weights = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inhibition = torch.sigmoid(torch.matmul(x, self.inhibition_weights))
        return x * (1 - inhibition)

class ShortTermMemory(nn.Module):
    """Short-term working memory for immediate context processing"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Attention mechanism for context processing
        self.context_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Temporal processing with GRU
        self.temporal_gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Context integration
        self.context_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Working memory state
        self.working_memory = None
        self.memory_capacity = 1000  # Number of recent items to maintain
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through working memory"""
        # Handle 4D input [B, C, H, W]
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Update working memory
        if self.working_memory is None:
            self.working_memory = x_flat
        else:
            # Concatenate new input with existing memory
            self.working_memory = torch.cat([self.working_memory, x_flat], dim=1)
            # Keep only recent items
            if self.working_memory.size(1) > self.memory_capacity:
                self.working_memory = self.working_memory[:, -self.memory_capacity:, :]
        
        # Apply self-attention for context processing
        context_out, _ = self.context_attention(x_flat, self.working_memory, self.working_memory)
        
        # Temporal processing
        temporal_out, _ = self.temporal_gru(context_out)
        
        # Combine context and temporal features
        combined = torch.cat([context_out, temporal_out], dim=-1)
        output = self.context_mlp(combined)
        
        # Reshape back to 4D
        output = output + x_flat  # Residual connection
        output = output.transpose(1, 2).reshape(B, C, H, W)
        
        return output

class LongTermMemory(nn.Module):
    """Long-term memory for storing and retrieving learned patterns"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Memory embeddings
        self.num_memory_slots = 10000
        self.memory_dim = config.hidden_dim
        self.memory_bank = nn.Parameter(
            torch.randn(self.num_memory_slots, self.memory_dim)
        )
        
        # Memory attention
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Memory update network
        self.memory_update = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Pattern completion network
        self.pattern_completion = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        # Initialize memory bank with orthogonal patterns
        nn.init.orthogonal_(self.memory_bank)
        
    def store(self, pattern: torch.Tensor):
        """Store pattern in memory bank"""
        # Ensure pattern is in full precision for storage
        pattern = pattern.to(dtype=self.memory_bank.dtype)
        indices = torch.arange(len(pattern))
        self.memory_bank.data[indices] = pattern.detach()
    
    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve similar patterns from memory"""
        B, HW, C = query.shape
        
        # Compute attention weights
        retrieved, _ = self.memory_attention(
            query,
            self.memory_bank.unsqueeze(0).expand(B, -1, -1),
            self.memory_bank.unsqueeze(0).expand(B, -1, -1)
        )
        
        # Complete partial patterns
        completed = self.pattern_completion(retrieved)
        
        # Combine with query through residual connection
        return completed + query
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through long-term memory"""
        # Handle 4D input [B, C, H, W]
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Store batch average pattern
        self.store(x_flat.mean(dim=1))
        
        # Retrieve and complete patterns
        retrieved = self.retrieve(x_flat)
        
        # Update memory representation
        combined = torch.cat([x_flat, retrieved], dim=-1)
        updated = self.memory_update(combined)
        
        # Reshape back to 4D
        output = updated.transpose(1, 2).reshape(B, C, H, W)
        
        return output

class DigitalNeocortex(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Enhanced pattern processor with multi-scale learning
        self.pattern_processor = MultiScalePatternProcessor(config)
        
        # Memory systems with increased capacity
        self.short_term_memory = ShortTermMemory(config)
        self.long_term_memory = LongTermMemory(config)
        
        # Enhanced cortical columns
        self.columns = nn.ModuleList([
            CorticalColumn(config) for _ in range(6)  # Increased from 4 to 6 columns
        ])
        
        # Multi-head cross-column attention
        self.column_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=16,  # Increased from 8 to 16 heads
            dropout=0.1,
            batch_first=True
        )
        
        # Hierarchical feature integration
        self.feature_integration = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(config.hidden_dim * (i+1), config.hidden_dim, 1),
                nn.LayerNorm([config.hidden_dim, 32, 32]),
                nn.GELU()
            ) for i in range(len(self.columns))
        ])
        
        # Enhanced output processor with perceptual features
        self.output_processor = nn.Sequential(
            # Multi-scale refinement
            nn.Conv2d(config.hidden_dim, config.hidden_dim * 2, 3, padding=1),
            nn.LayerNorm([config.hidden_dim * 2, 32, 32]),
            nn.GELU(),
            
            # Perceptual feature extraction
            nn.Conv2d(config.hidden_dim * 2, config.hidden_dim * 2, 3, padding=1, groups=2),
            nn.LayerNorm([config.hidden_dim * 2, 32, 32]),
            nn.GELU(),
            
            # Channel attention
            ChannelAttention(config.hidden_dim * 2),
            
            # Final refinement
            nn.Conv2d(config.hidden_dim * 2, config.hidden_dim, 3, padding=1),
            nn.LayerNorm([config.hidden_dim, 32, 32]),
            nn.GELU(),
            
            # Final output
            nn.Conv2d(config.hidden_dim, 1, 1)  # Output single channel
        )
        
        # Error feedback mechanism
        self.error_processor = nn.Sequential(
            nn.Conv2d(config.hidden_dim, config.hidden_dim * 2, 3, padding=1),
            nn.LayerNorm([config.hidden_dim * 2, 32, 32]),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim * 2, config.hidden_dim, 1)
        )
        
        self.prev_errors = None
    
    def forward(self, x: torch.Tensor, modality: str = 'pattern') -> torch.Tensor:
        """Process input through the enhanced neocortex"""
        batch_size = x.shape[0]
        
        # Ensure input has proper dimensions [B, C, H, W]
        if len(x.shape) == 2:
            seq_len = x.shape[1]
            spatial_size = int(np.sqrt(seq_len))
            x = x.view(batch_size, 1, spatial_size, spatial_size)
        elif len(x.shape) == 3:
            seq_len = x.shape[1]
            spatial_size = int(np.sqrt(seq_len))
            x = x.transpose(1, 2)
            x = x.reshape(batch_size, -1, spatial_size, spatial_size)
        
        # Multi-scale pattern processing with error feedback
        x, current_errors = self.pattern_processor(x, self.prev_errors)
        self.prev_errors = current_errors
        
        # Process through enhanced cortical columns with progressive feature integration
        column_outputs = []
        integrated_features = x
        
        for i, column in enumerate(self.columns):
            # Get lateral input from previous timestep
            lateral_input = column_outputs[i-1] if i > 0 else None
            
            # Process through column
            col_out, _ = column(integrated_features, lateral_input)
            column_outputs.append(col_out)
            
            # Integrate features progressively
            stacked_outputs = torch.cat(column_outputs, dim=1)
            integrated_features = self.feature_integration[i](stacked_outputs)
        
        # Cross-column attention for global coherence
        B, C, H, W = integrated_features.shape
        flat_features = integrated_features.flatten(2).transpose(1, 2)
        attended_features, _ = self.column_attention(flat_features, flat_features, flat_features)
        integrated_features = attended_features.transpose(1, 2).reshape(B, C, H, W)
        
        # Update memory systems with enhanced features
        stm_out = self.short_term_memory(integrated_features)
        ltm_out = self.long_term_memory(stm_out)
        
        # Process error feedback
        error_signal = self.error_processor(ltm_out - x)
        output = ltm_out + error_signal
        
        # Final processing
        output = self.output_processor(output)
        
        if modality == 'pattern':
            return output  # Already in [B, C=1, H, W] format
        
        return output

    def get_attention_weights(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract attention weights from all layers"""
        attention_weights = {}
        
        # Process input and collect attention weights
        with torch.no_grad():
            # Store original hooks
            original_hooks = {}
            for name, module in self.named_modules():
                if hasattr(module, 'attention_weights'):
                    original_hooks[name] = module._forward_hooks
            
            # Register hooks to capture attention weights
            hooks = []
            def get_attention_hook(name):
                def hook(module, input, output):
                    if hasattr(module, 'attention_weights'):
                        attention_weights[name] = module.attention_weights.detach()
                return hook
            
            for name, module in self.named_modules():
                if hasattr(module, 'attention_weights'):
                    hooks.append(module.register_forward_hook(get_attention_hook(name)))
            
            # Forward pass to collect weights
            _ = self(x, modality='pattern')
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Restore original hooks
            for name, module in self.named_modules():
                if name in original_hooks:
                    module._forward_hooks = original_hooks[name]
        
        return attention_weights

class CorticalColumn(nn.Module):
    """Hierarchical processing unit inspired by neocortical columns"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initial downsampling to 8x8
        self.downsample = nn.Sequential(
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, stride=2, padding=1),
            nn.LayerNorm([config.hidden_dim, 16, 16]),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, stride=2, padding=1),
            nn.LayerNorm([config.hidden_dim, 8, 8]),
            nn.GELU()
        )
        
        # Hierarchical convolutional layers
        self.layers = nn.ModuleList([
            # Layer 1: Feature extraction
            nn.Sequential(
                nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
                nn.LayerNorm([config.hidden_dim, 8, 8]),
                nn.GELU()
            ),
            # Layer 2: Pattern recognition
            nn.Sequential(
                nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
                nn.LayerNorm([config.hidden_dim, 8, 8]),
                nn.GELU(),
                nn.Dropout2d(0.1)
            ),
            # Layer 3: Abstract reasoning
            nn.Sequential(
                nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
                nn.LayerNorm([config.hidden_dim, 8, 8]),
                nn.GELU(),
                nn.Dropout2d(0.2)
            )
        ])
        
        # Lateral connections between columns
        self.lateral_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Top-down modulation
        self.top_down = nn.Sequential(
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
            nn.LayerNorm([config.hidden_dim, 8, 8]),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 1)
        )
        
        # Bottom-up integration
        self.bottom_up = nn.Sequential(
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
            nn.LayerNorm([config.hidden_dim, 8, 8]),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 1)
        )
        
        # Predictive coding
        self.prediction = nn.Sequential(
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
            nn.LayerNorm([config.hidden_dim, 8, 8]),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 1)
        )
        
        # Upsample back to original size
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_dim, config.hidden_dim, 4, stride=2, padding=1),
            nn.LayerNorm([config.hidden_dim, 16, 16]),
            nn.GELU(),
            nn.ConvTranspose2d(config.hidden_dim, config.hidden_dim, 4, stride=2, padding=1),
            nn.LayerNorm([config.hidden_dim, 32, 32]),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor, lateral_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process input through cortical column with hierarchical processing"""
        # Downsample input to 8x8
        x = self.downsample(x)
        current = x
        
        # Process through hierarchical layers
        for layer in self.layers:
            current = layer(current)
            
            # Apply lateral connections if available
            if lateral_input is not None:
                B, C, H, W = current.shape
                current_flat = current.flatten(2).transpose(1, 2)  # [B, H*W, C]
                lateral_flat = lateral_input.flatten(2).transpose(1, 2)  # [B, H*W, C]
                
                # Apply lateral attention
                attended, _ = self.lateral_attention(current_flat, lateral_flat, lateral_flat)
                lateral_out = attended.transpose(1, 2).reshape(B, C, H, W)
                
                # Combine with current features
                current = current + 0.1 * lateral_out
        
        # Generate prediction
        prediction = self.prediction(current)
        
        # Apply top-down and bottom-up processing
        top_down = self.top_down(current)
        bottom_up = self.bottom_up(current)
        
        # Combine processing streams
        output = current + 0.1 * top_down + 0.1 * bottom_up
        
        # Upsample back to original size
        output = self.upsample(output)
        prediction = self.upsample(prediction)
        
        return output, prediction

class HierarchicalLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Attention
        self.attention = MultiHeadAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=0.1
        )
        
        # Feed-forward with positive correlation
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU()
        )
        
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x):
        # Store original shape
        orig_shape = x.shape
        
        # Reshape if needed for attention
        if len(x.shape) == 4:  # [batch, h, w, dim]
            b, h, w, d = x.shape
            x = x.view(b, h*w, d)
        
        # Process with attention
        attended = self.attention(self.norm1(x))
        
        # Ensure attended output matches input shape
        if attended.shape != x.shape:
            attended = attended[:, :x.shape[1], :]
            
        x = x + torch.abs(self.residual_scale) * attended
        
        # Process with feed-forward
        ff_output = self.feed_forward(self.norm2(x))
        x = x + torch.abs(self.residual_scale) * ff_output
        
        # Restore original shape if needed
        if len(orig_shape) == 4:
            x = x.view(orig_shape)
            
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Ensure positive attention
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Convert attention weights to numpy immediately
        self.last_attn_weights = attn.detach().cpu().numpy()
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        return self.o_proj(out)

class PredictiveCoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)

class LocalInhibitionCircuit(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Inhibitory interneurons
        self.inhibitory_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate inhibitory signals
        inhibition = self.inhibitory_layer(x)
        # Apply lateral inhibition
        return x * (1 - inhibition)

class TemporalContextAnalyzer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.temporal_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.temporal_net(x)

class SemanticContextAnalyzer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.semantic_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(), 
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.semantic_net(x)

class TemporalProcessor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Temporal memory gate
        self.temporal_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Pattern memory
        self.pattern_memory = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Direction gate to fix inverse relationships
        self.direction_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Process temporal relationships
        h_0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        pattern_features, _ = self.pattern_memory(x, h_0)
        
        # Compute temporal context
        temporal_context = torch.cat([x, pattern_features], dim=-1)
        temporal_gate = self.temporal_gate(temporal_context)
        
        # Apply direction correction
        direction = self.direction_gate(pattern_features)
        
        # Combine with original input
        output = x * temporal_gate * direction
        
        return output

class PatternMatcher(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Multi-scale pattern recognition
        self.pattern_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ) for _ in range(3)  # 3 different scales
        ])
        
        # Scale-aware alignment
        self.align = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # 3 scales
            nn.Softmax(dim=-1)
        )
        
        # Pattern enhancement with skip connection
        self.enhance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, x):
        # Multi-scale pattern processing
        patterns = [proj(x) for proj in self.pattern_proj]
        pattern_cat = torch.cat([p.mean(dim=1) for p in patterns], dim=-1)
        
        # Compute scale-aware alignment
        weights = self.align(pattern_cat).unsqueeze(1)
        
        # Combine patterns at different scales
        combined_pattern = sum(w * p for w, p in zip(weights.chunk(3, dim=-1), patterns))
        
        # Enhance patterns with residual connection
        enhanced = self.enhance(combined_pattern)
        return enhanced + x

class TestProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Enhanced input processing
        self.input_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.LayerNorm(config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )
        
        # Improved pattern matcher with residual connections
        self.pattern_matcher = PatternMatcher(config.hidden_dim)
        
        # Enhanced output processing
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        
        identity = x
        
        # Multi-stage processing with residual connection
        x = self.input_proj(x)
        x = self.pattern_matcher(x)
        x = x + identity
        
        return x
    
    def post_process(self, x):
        return self.output_proj(x)

class SpatialAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Multi-scale spatial attention
        self.query_conv = nn.Conv2d(hidden_dim, hidden_dim // 8, 1)
        self.key_conv = nn.Conv2d(hidden_dim, hidden_dim // 8, 1)
        self.value_conv = nn.Conv2d(hidden_dim, hidden_dim, 1)
        
        # Position-aware encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, hidden_dim, 32, 32))
        
        # Output projection
        self.output_conv = nn.Conv2d(hidden_dim, hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        # Add positional information
        x = x + self.pos_embedding[:, :, :h, :w]
        
        # Compute attention maps
        query = self.query_conv(x).view(b, -1, h * w)
        key = self.key_conv(x).view(b, -1, h * w)
        value = self.value_conv(x).view(b, -1, h * w)
        
        # Compute attention scores
        attention = F.softmax(torch.bmm(query.transpose(1, 2), key), dim=-1)
        
        # Apply attention
        out = torch.bmm(value, attention.transpose(1, 2))
        out = out.view(b, -1, h, w)
        
        return self.output_conv(out)

class SpatialProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Multi-scale feature extraction
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
                nn.BatchNorm2d(config.hidden_dim),
                nn.GELU(),
                nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
                nn.BatchNorm2d(config.hidden_dim),
                nn.GELU()
            ) for _ in range(3)  # 3 different scales
        ])
        
        # Scale attention
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(config.hidden_dim * 3, config.hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim, 3, 1),
            nn.Softmax(dim=1)
        )
        
        # Spatial attention
        self.spatial_attention = SpatialAttention(config.hidden_dim)
        
        # Feature enhancement
        self.enhance = nn.Sequential(
            nn.Conv2d(config.hidden_dim, config.hidden_dim * 2, 1),
            nn.BatchNorm2d(config.hidden_dim * 2),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim * 2, config.hidden_dim, 1),
            nn.BatchNorm2d(config.hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale processing
        features = []
        for conv in self.conv_layers:
            features.append(conv(x))
        
        # Concatenate features
        multi_scale = torch.cat(features, dim=1)
        
        # Compute scale attention
        scale_weights = self.scale_attention(multi_scale)
        
        # Apply scale attention
        weighted_features = sum(w * f for w, f in zip(
            scale_weights.chunk(3, dim=1),
            features
        ))
        
        # Apply spatial attention
        spatial_features = self.spatial_attention(weighted_features)
        
        # Enhance features
        enhanced = self.enhance(spatial_features)
        
        # Residual connection
        return enhanced + x

class PatternProcessor(nn.Module):
    """Initial processing of input patterns"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initial channel expansion with better preservation
        self.channel_expand = nn.Sequential(
            nn.Conv2d(1, config.hidden_dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim // 2, config.hidden_dim, 1)
        )
        
        # Multi-scale feature extraction with skip connections
        self.encoder1 = nn.Sequential(
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
            nn.LayerNorm([config.hidden_dim, 32, 32]),
            nn.GELU()
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, stride=2, padding=1),
            nn.LayerNorm([config.hidden_dim, 16, 16]),
            nn.GELU()
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, stride=2, padding=1),
            nn.LayerNorm([config.hidden_dim, 8, 8]),
            nn.GELU()
        )
        
        # Decoder path with skip connections
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_dim, config.hidden_dim, 4, stride=2, padding=1),
            nn.LayerNorm([config.hidden_dim, 16, 16]),
            nn.GELU()
        )
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_dim, config.hidden_dim, 4, stride=2, padding=1),
            nn.LayerNorm([config.hidden_dim, 32, 32]),
            nn.GELU()
        )
        
        # Multi-head self-attention for global context
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(config.hidden_dim * 3, config.hidden_dim, 1),
            nn.LayerNorm([config.hidden_dim, 32, 32]),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
            nn.LayerNorm([config.hidden_dim, 32, 32]),
            nn.GELU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input pattern with multi-scale features and skip connections"""
        # Expand channels if needed
        if x.shape[1] == 1:
            x = self.channel_expand(x)
        
        # Encoder path with skip connections
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        # Apply attention at the bottleneck
        B, C, H, W = e3.shape
        x_flat = e3.flatten(2).transpose(1, 2)  # [B, H*W, C]
        attended, _ = self.attention(x_flat, x_flat, x_flat)
        e3 = attended.transpose(1, 2).reshape(B, C, H, W)
        
        # Decoder path with skip connections
        d2 = self.decoder2(e3)
        d1 = self.decoder1(d2)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([
            d1,  # Full resolution
            F.interpolate(d2, size=d1.shape[2:], mode='bilinear', align_corners=False),
            F.interpolate(e3, size=d1.shape[2:], mode='bilinear', align_corners=False)
        ], dim=1)
        
        # Final refinement
        out = self.refine(multi_scale)
        
        return out

class MultiScalePatternProcessor(nn.Module):
    """Advanced pattern processor with multi-scale learning and feedback loops"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        # 2D Pattern Detection
        self.pattern_detectors = nn.ModuleDict({
            'checkerboard': nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.LayerNorm([32, 32, 32]),
                nn.GELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=8),
                nn.LayerNorm([32, 32, 32]),
                nn.GELU()
            ),
            'radial': nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, padding=2),
                nn.LayerNorm([32, 32, 32]),
                nn.GELU(),
                nn.Conv2d(32, 32, kernel_size=5, padding=2, groups=8),
                nn.LayerNorm([32, 32, 32]),
                nn.GELU()
            )
        })

        # Initialize 2D pattern kernels
        with torch.no_grad():
            # Pattern-matching checkerboard kernel
            checker = torch.tensor([
                [-2.5, 2.5, -2.5],
                [2.5, -4, 2.5],  # Strong center for exact matching
                [-2.5, 2.5, -2.5]
            ]).float().view(1, 1, 3, 3)
            
            # Initialize multiple rotated versions for better pattern matching
            checker_90 = torch.rot90(checker.view(3, 3), k=1).view(1, 1, 3, 3)
            checker_45 = torch.tensor([
                [0, -2.5, 0],
                [-2.5, 4, -2.5],
                [0, -2.5, 0]
            ]).float().view(1, 1, 3, 3)
            
            # Combine different orientations
            self.pattern_detectors['checkerboard'][0].weight.data[:8] = checker.repeat(8, 1, 1, 1)
            self.pattern_detectors['checkerboard'][0].weight.data[8:16] = checker_90.repeat(8, 1, 1, 1)
            self.pattern_detectors['checkerboard'][0].weight.data[16:24] = checker_45.repeat(8, 1, 1, 1)
            self.pattern_detectors['checkerboard'][0].weight.data[24:] = -checker_45.repeat(8, 1, 1, 1)

            # Initialize radial attention with pattern-focused falloff
            radial_dist = torch.zeros(32, 32)
            center_y, center_x = 15.5, 15.5
            y_coords = torch.arange(32).float().view(-1, 1).repeat(1, 32)
            x_coords = torch.arange(32).float().view(1, -1).repeat(32, 1)
            dist = torch.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
            
            # Create a more pattern-focused attention map
            radial_dist = torch.exp(-dist / 5.0) * (dist < 12).float() * 1.2  # Sharp cutoff at pattern boundary
            
            self.radial_attention = nn.Parameter(radial_dist.view(1, 1, 32, 32))

        # Pattern-focused processing
        self.pattern_enhance = nn.ModuleDict({
            scale: nn.Sequential(
                nn.Conv2d(64, self.hidden_dim, 1),  # 64 = pattern channels
                nn.LayerNorm([self.hidden_dim, size, size]),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, groups=8),
                nn.LayerNorm([self.hidden_dim, size, size]),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
            ) for scale, size in [('32x32', 32), ('16x16', 16), ('8x8', 8)]
        })

        # Pattern attention modules for each scale
        self.pattern_attention = nn.ModuleDict({
            scale: nn.Sequential(
                nn.Conv2d(self.hidden_dim, 64, 1),
                nn.LayerNorm([64, size, size]),
                nn.GELU(),
                nn.Conv2d(64, 1, 1),
                nn.Sigmoid()
            ) for scale, size in [('32x32', 32), ('16x16', 16), ('8x8', 8)]
        })

        # Edge detection kernels for sharp transitions
        self.edge_detectors = nn.ModuleDict({
            'vertical': nn.Conv2d(1, 16, kernel_size=(3, 1), padding=(1, 0), bias=False),
            'horizontal': nn.Conv2d(1, 16, kernel_size=(1, 3), padding=(0, 1), bias=False),
            'diagonal': nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        })
        
        # Initialize edge detection kernels
        with torch.no_grad():
            # Vertical edge detection
            self.edge_detectors['vertical'].weight.data[:8] = torch.tensor([
                [-1, 0, 1],
            ]).float().view(1, 1, 3, 1).repeat(8, 1, 1, 1)
            self.edge_detectors['vertical'].weight.data[8:] = torch.tensor([
                [1, 0, -1],
            ]).float().view(1, 1, 3, 1).repeat(8, 1, 1, 1)
            
            # Horizontal edge detection
            self.edge_detectors['horizontal'].weight.data[:8] = torch.tensor([
                [-1],
                [0],
                [1],
            ]).float().view(1, 1, 1, 3).repeat(8, 1, 1, 1)
            self.edge_detectors['horizontal'].weight.data[8:] = torch.tensor([
                [1],
                [0],
                [-1],
            ]).float().view(1, 1, 1, 3).repeat(8, 1, 1, 1)

        # Rhythm detection module
        self.rhythm_detector = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 5), padding=(0, 2)),  # Detect 5-pixel patterns
            nn.LayerNorm([32, 32, 32]),
            nn.GELU(),
            nn.Conv2d(32, self.hidden_dim, 1)
        )

        # Sharp feature enhancement
        self.sharp_enhance = nn.ModuleDict({
            scale: nn.Sequential(
                nn.Conv2d(self.hidden_dim + 48 + 64, self.hidden_dim, 1),  # 48 = 16*3 edge channels, 64 = 2*32 pattern channels
                nn.LayerNorm([self.hidden_dim, size, size]),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, groups=8),
                nn.LayerNorm([self.hidden_dim, size, size]),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
            ) for scale, size in [('32x32', 32), ('16x16', 16), ('8x8', 8)]
        })

        # Smoothing module
        self.smooth_enhance = nn.ModuleDict({
            scale: nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim, 5, padding=2, groups=self.hidden_dim),
                nn.LayerNorm([self.hidden_dim, size, size]),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, 1),
                nn.LayerNorm([self.hidden_dim, size, size]),
                nn.GELU()
            ) for scale, size in [('32x32', 32), ('16x16', 16), ('8x8', 8)]
        })

        # Adaptive feature mixing
        self.feature_mix = nn.ModuleDict({
            scale: nn.Sequential(
                nn.Conv2d(self.hidden_dim * 2, 2, 1),  # 2 channels for mixing weights
                nn.Softmax(dim=1)
            ) for scale, size in [('32x32', 32), ('16x16', 16), ('8x8', 8)]
        })

        # Detail preservation layers
        self.detail_preserve = nn.ModuleDict({
            '32x32': nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.LayerNorm([32, 32, 32]),
                nn.GELU(),
                nn.Conv2d(32, self.hidden_dim, 1)
            ),
            '16x16': nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.LayerNorm([32, 16, 16]),
                nn.GELU(),
                nn.Conv2d(32, self.hidden_dim, 1)
            ),
            '8x8': nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.LayerNorm([32, 8, 8]),
                nn.GELU(),
                nn.Conv2d(32, self.hidden_dim, 1)
            )
        })

        # Multi-scale encoders with residual connections
        self.encoders = nn.ModuleDict({
            '32x32': self._make_encoder_block(32),
            '16x16': self._make_encoder_block(16),
            '8x8': self._make_encoder_block(8)
        })

        # Enhanced error prediction
        self.error_predictor = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, 3, padding=1),
            nn.LayerNorm([self.hidden_dim * 2, 32, 32]),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, 1),
            nn.Sigmoid()
        )

        # Dimension reduction with residual
        self.dim_reduce = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 3, self.hidden_dim * 2, 1),
            nn.LayerNorm([self.hidden_dim * 2, 32, 32]),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, 1)
        )

        # Cross-scale attention with gradient checkpointing
        self.cross_scale_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Enhanced feature fusion with residual
        self.fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, 3, padding=1),
                nn.LayerNorm([self.hidden_dim * 2, 32, 32]),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, 1)
            ) for _ in range(2)
        ])

        # Error gate with learned temperature
        self.error_gate = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, 1),
            nn.LayerNorm([self.hidden_dim, 32, 32]),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 1),
            nn.Sigmoid()
        )
        self.error_temperature = nn.Parameter(torch.ones(1))

    def _make_encoder_block(self, size):
        return nn.ModuleDict({
            'main': nn.Sequential(
                nn.Conv2d(1, self.hidden_dim // 2, 3, padding=1),
                nn.LayerNorm([self.hidden_dim // 2, size, size]),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim // 2, self.hidden_dim, 3, padding=1),
                nn.LayerNorm([self.hidden_dim, size, size]),
                nn.GELU()
            ),
            'residual': nn.Sequential(
                nn.Conv2d(1, self.hidden_dim, 1),
                nn.LayerNorm([self.hidden_dim, size, size])
            )
        })

    def forward(self, x, prev_errors=None):
        batch_size = x.shape[0]
        
        # Ensure input has correct channel dimension
        if x.dim() == 2:  # [B, T]
            x = x.view(batch_size, 1, 32, 32)
        elif x.dim() == 3:  # [B, H, W]
            x = x.unsqueeze(1)  # Add channel dimension
        elif x.dim() == 4 and x.shape[1] > 1:  # [B, C>1, H, W]
            x = x.mean(dim=1, keepdim=True)  # Average channels

        # Detect 2D patterns
        checker_features = self.pattern_detectors['checkerboard'](x)
        radial_features = self.pattern_detectors['radial'](x)
        
        # Detect edges and rhythmic patterns
        edges_v = self.edge_detectors['vertical'](x)
        edges_h = self.edge_detectors['horizontal'](x)
        edges_d = self.edge_detectors['diagonal'](x)
        rhythm = self.rhythm_detector(x)

        # Process at different scales
        features_32 = self._process_scale(x, '32x32', checker_features, radial_features, edges_v, edges_h, edges_d, rhythm)
        features_16 = self._process_scale(
            F.interpolate(x, size=(16, 16), mode='bicubic', align_corners=False),
            '16x16',
            F.interpolate(checker_features, size=(16, 16), mode='bicubic', align_corners=False),
            F.interpolate(radial_features, size=(16, 16), mode='bicubic', align_corners=False),
            F.interpolate(edges_v, size=(16, 16), mode='bicubic', align_corners=False),
            F.interpolate(edges_h, size=(16, 16), mode='bicubic', align_corners=False),
            F.interpolate(edges_d, size=(16, 16), mode='bicubic', align_corners=False),
            F.interpolate(rhythm, size=(16, 16), mode='bicubic', align_corners=False)
        )
        features_8 = self._process_scale(
            F.interpolate(x, size=(8, 8), mode='bicubic', align_corners=False),
            '8x8',
            F.interpolate(checker_features, size=(8, 8), mode='bicubic', align_corners=False),
            F.interpolate(radial_features, size=(8, 8), mode='bicubic', align_corners=False),
            F.interpolate(edges_v, size=(8, 8), mode='bicubic', align_corners=False),
            F.interpolate(edges_h, size=(8, 8), mode='bicubic', align_corners=False),
            F.interpolate(edges_d, size=(8, 8), mode='bicubic', align_corners=False),
            F.interpolate(rhythm, size=(8, 8), mode='bicubic', align_corners=False)
        )

        # Detail-preserving upsampling
        features_16 = F.interpolate(features_16, size=(32, 32), mode='bicubic', align_corners=False)
        features_8 = F.interpolate(features_8, size=(32, 32), mode='bicubic', align_corners=False)

        # Concatenate and reduce dimension with residual
        features = torch.cat([features_32, features_16, features_8], dim=1)
        features = self.dim_reduce(features) + features_32  # Residual from highest resolution

        # Cross-scale attention with gradient checkpointing
        B, C, H, W = features.shape
        features_flat = features.flatten(2).transpose(1, 2)
        if self.training:
            attended_features, _ = torch.utils.checkpoint.checkpoint(
                self.cross_scale_attention, features_flat, features_flat, features_flat,
                use_reentrant=False
            )
        else:
            attended_features, _ = self.cross_scale_attention(features_flat, features_flat, features_flat)
        features = attended_features.transpose(1, 2).reshape(B, C, H, W)

        # Multi-stage feature fusion with residuals
        for fusion_layer in self.fusion:
            features = features + fusion_layer(features)

        # Enhanced error feedback
        current_errors = None
        if prev_errors is not None:
            # Ensure prev_errors has correct channel dimension
            if prev_errors.shape[1] > 1:
                prev_errors = prev_errors.mean(dim=1, keepdim=True)
                
            # Predict error importance
            error_importance = self.error_predictor(prev_errors)
            
            # Temperature-scaled error gating
            error_gate = self.error_gate(torch.cat([features, prev_errors], dim=1))
            error_gate = error_gate * torch.sigmoid(self.error_temperature)
            
            # Apply error feedback
            features = features * (1 + error_gate * error_importance)
            current_errors = features - x

        return features, current_errors
    
    def _process_scale(self, x, scale, checker_features, radial_features, edges_v, edges_h, edges_d, rhythm):
        # Detail preservation with pattern focus
        details = self.detail_preserve[scale](x)
        
        # Multi-orientation pattern processing
        pattern_features = torch.cat([
            checker_features[:, :8] * 1.3,  # Original orientation
            checker_features[:, 8:16] * 1.3,  # 90 degree rotation
            checker_features[:, 16:] * 1.2,  # 45 degree patterns
            radial_features
        ], dim=1)
        pattern_enhanced = self.pattern_enhance[scale](pattern_features)
        
        # Apply pattern-focused attention
        size = int(scale.split('x')[0])
        if size == 32:
            pattern_enhanced = pattern_enhanced * (self.radial_attention ** 0.8)
        else:
            scaled_attention = F.interpolate(self.radial_attention, size=(size, size), mode='bilinear', align_corners=True)
            pattern_enhanced = pattern_enhanced * (scaled_attention ** 0.8)
        
        # Main processing with pattern preservation
        encoder = self.encoders[scale]
        features = encoder['main'](x) + encoder['residual'](x) * 1.2
        
        # Edge-pattern combination
        edge_features = torch.cat([edges_v * 1.2, edges_h * 1.2, edges_d], dim=1)
        features = torch.cat([features, edge_features, pattern_features], dim=1)
        features = self.sharp_enhance[scale](features)
        
        # Pattern-aware attention with orientation sensitivity
        pattern_weights = self.pattern_attention[scale](pattern_enhanced)
        features = features * (1 + pattern_weights * 1.2)
        
        # Reduced smoothing for pattern preservation
        smooth_features = self.smooth_enhance[scale](features) * 0.85
        
        # Adaptive mixing favoring pattern structure
        combined = torch.cat([features, smooth_features], dim=1)
        mix_weights = self.feature_mix[scale](combined)
        features = features * mix_weights[:, 0:1] * 1.15 + smooth_features * mix_weights[:, 1:2] * 0.85
        
        # Final combination emphasizing pattern replication
        return features + 0.35 * pattern_enhanced + 0.25 * details + 0.15 * rhythm

class ChannelAttention(nn.Module):  
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention = self.avg_pool(x)
        attention = self.attention(attention)
        return x * attention