import torch
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class TrainingConfig:
    # Neural network architecture parameters
    input_dim: int = 64  # Default input dimension
    hidden_dim: int = 128  # Increased hidden dimension for better capacity
    num_layers: int = 6  # More layers for hierarchical processing
    num_columns: int = 4  # More cortical columns for parallel processing
    memory_capacity: int = 256  # Increased memory for better pattern retention
    num_heads: int = 8  # More attention heads for multi-scale features
    dropout: float = 0.15  # Slightly increased dropout for regularization
    action_dim: int = 16  # Increased action space
    sequence_length: int = 32  # Sequence length
    
    # Spatial pattern specific parameters
    spatial_kernel_size: int = 3  # Size of spatial convolution kernel
    spatial_channels: int = 16  # Number of spatial feature channels
    use_spatial_attention: bool = True  # Enable spatial attention
    spatial_pooling_type: str = 'adaptive'  # Type of spatial pooling
    
    # Training hyperparameters
    max_epochs: int = 200  # More epochs for better convergence
    batch_size: int = 8  # Increased batch size
    initial_lr: float = 0.002  # Slightly higher learning rate
    min_lr: float = 1e-6
    gradient_clip: float = 0.5  # More aggressive gradient clipping
    early_stop_patience: int = 10  # More patience for complex patterns
    validation_split: float = 0.2
    
    # Advanced training options
    use_curriculum: bool = True  # Enable curriculum learning
    use_focal_loss: bool = True  # Enable focal loss for hard patterns
    focal_gamma: float = 2.0  # Focal loss gamma parameter
    use_mixup: bool = True  # Enable mixup augmentation
    mixup_alpha: float = 0.2  # Mixup interpolation strength
    
    # Pattern-specific loss weights
    pattern_weights: Dict[str, float] = None
    
    # Neural network comparison parameters
    is_spiking: bool = False
    learning_rate: float = 0.002
    
    def __post_init__(self):
        if self.pattern_weights is None:
            # Balanced weights for different pattern aspects
            self.pattern_weights = {
                'sequence': 1.0,
                'patterns': 1.0,
                'mixed': 1.0,
                'temporal': 1.0,
                'spatial': 1.5,  # Higher weight for spatial patterns
                'structure': 1.2,  # New weight for structural similarity
                'gradient': 1.2,  # New weight for gradient matching
                'frequency': 1.1  # New weight for frequency content
            }

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop

class MetricsTracker:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.pattern_losses = defaultdict(list)
        self.attention_stats = defaultdict(list)
        self.memory_stats = defaultdict(list)
        self.layer_stats = defaultdict(list)
        self.spatial_stats = defaultdict(list)  # New tracker for spatial metrics
        
    def update_pattern_loss(self, pattern_type: str, loss: float):
        self.pattern_losses[pattern_type].append(loss)
        
    def update_attention_stats(self, attention: torch.Tensor):
        # attention shape: [batch, heads, seq_len, seq_len]
        
        # Compute per-head metrics
        for head in range(attention.size(1)):
            head_attention = attention[:, head]  # [batch, seq_len, seq_len]
            
            stats = {
                'entropy': self._compute_attention_entropy(head_attention),
                'sparsity': self._compute_attention_sparsity(head_attention),
                'focus': self._compute_attention_focus(head_attention),
                'spatial_coherence': self._compute_spatial_coherence(head_attention)  # New metric
            }
            
            # Store per-head stats
            for key, value in stats.items():
                self.attention_stats[f'head_{head}_{key}'].append(value)
    
    def update_spatial_stats(self, output: torch.Tensor, target: torch.Tensor):
        """Track spatial reconstruction quality metrics"""
        # Compute structural similarity
        ssim = self._compute_ssim(output, target)
        self.spatial_stats['ssim'].append(ssim.item())
        
        # Compute gradient matching
        grad_similarity = self._compute_gradient_similarity(output, target)
        self.spatial_stats['gradient_similarity'].append(grad_similarity.item())
        
        # Compute frequency domain similarity
        freq_similarity = self._compute_frequency_similarity(output, target)
        self.spatial_stats['frequency_similarity'].append(freq_similarity.item())
    
    @staticmethod
    def _compute_spatial_coherence(attention: torch.Tensor) -> float:
        """Compute spatial coherence of attention patterns"""
        # Compute local structure similarity
        attention = attention.mean(0)  # Average over batch
        kernel = torch.ones(3, 3) / 9.0  # 3x3 averaging kernel
        kernel = kernel.to(attention.device)
        
        # Pad attention map
        padded = torch.nn.functional.pad(attention, (1, 1, 1, 1), mode='reflect')
        
        # Compute local averages
        local_avg = torch.nn.functional.conv2d(
            padded.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0)
        ).squeeze()
        
        # Compute coherence as correlation between original and local average
        coherence = torch.corrcoef(attention.flatten(), local_avg.flatten())[0, 1]
        return coherence.item()
    
    @staticmethod
    def _compute_ssim(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Structural Similarity Index (SSIM)"""
        # Constants for stability
        C1 = (0.01 * 2) ** 2
        C2 = (0.03 * 2) ** 2
        
        # Compute means
        mu_x = torch.mean(output, dim=(-2, -1), keepdim=True)
        mu_y = torch.mean(target, dim=(-2, -1), keepdim=True)
        
        # Compute variances and covariance
        sigma_x = torch.var(output, dim=(-2, -1), keepdim=True, unbiased=False)
        sigma_y = torch.var(target, dim=(-2, -1), keepdim=True, unbiased=False)
        sigma_xy = torch.mean((output - mu_x) * (target - mu_y), dim=(-2, -1), keepdim=True)
        
        # Compute SSIM
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        
        return ssim.mean()
    
    @staticmethod
    def _compute_gradient_similarity(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute gradient-based similarity"""
        # Compute gradients
        output_dx = output[..., 1:] - output[..., :-1]
        output_dy = output[..., 1:, :] - output[..., :-1, :]
        target_dx = target[..., 1:] - target[..., :-1]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        
        # Compute cosine similarity between gradients
        sim_x = torch.nn.functional.cosine_similarity(output_dx, target_dx, dim=-1).mean()
        sim_y = torch.nn.functional.cosine_similarity(output_dy, target_dy, dim=-1).mean()
        
        return (sim_x + sim_y) / 2
    
    @staticmethod
    def _compute_frequency_similarity(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute similarity in frequency domain"""
        # Compute 2D FFT
        output_fft = torch.fft.fft2(output)
        target_fft = torch.fft.fft2(target)
        
        # Compute magnitude spectra
        output_mag = torch.abs(output_fft)
        target_mag = torch.abs(target_fft)
        
        # Normalize and compute similarity
        output_mag_norm = output_mag / (output_mag.sum() + 1e-8)
        target_mag_norm = target_mag / (target_mag.sum() + 1e-8)
        
        return -torch.nn.functional.kl_div(
            torch.log(output_mag_norm + 1e-8),
            target_mag_norm,
            reduction='mean'
        )
    
    @staticmethod
    def _compute_attention_entropy(attention: torch.Tensor) -> float:
        # attention shape: [batch, seq_len, seq_len]
        attention = attention.mean(0)  # Average over batch
        # Add small epsilon to avoid log(0)
        attention = attention + 1e-10
        # Normalize
        attention = attention / attention.sum(dim=-1, keepdim=True)
        entropy = -torch.sum(attention * torch.log2(attention), dim=-1)
        return entropy.mean().item()
    
    @staticmethod
    def _compute_attention_sparsity(attention: torch.Tensor) -> float:
        # attention shape: [batch, seq_len, seq_len]
        attention = attention.mean(0)  # Average over batch
        # Consider values below threshold as zero
        sparsity = (attention < 0.02).float().mean().item()
        return sparsity
    
    @staticmethod
    def _compute_attention_focus(attention: torch.Tensor) -> float:
        # attention shape: [batch, seq_len, seq_len]
        attention = attention.mean(0)  # Average over batch
        # Maximum attention value per query
        focus = attention.max(dim=-1)[0].mean().item()
        return focus
            
    def update_memory_stats(self, memory: torch.Tensor):
        stats = {
            'utilization': self._compute_memory_utilization(memory),
            'sparsity': self._compute_memory_sparsity(memory),
            'coherence': self._compute_memory_coherence(memory)
        }
        for key, value in stats.items():
            self.memory_stats[key].append(value)
            
    def update_layer_stats(self, layer_outputs: List[torch.Tensor]):
        for i, layer_output in enumerate(layer_outputs):
            stats = {
                'mean_activation': layer_output.mean().item(),
                'std_activation': layer_output.std().item(),
                'sparsity': (layer_output > 0).float().mean().item()
            }
            self.layer_stats[f'layer_{i}'].append(stats)
    
    @staticmethod
    def _compute_memory_utilization(memory: torch.Tensor) -> float:
        return (torch.abs(memory) > 0.1).float().mean().item()
    
    @staticmethod
    def _compute_memory_sparsity(memory: torch.Tensor) -> float:
        return (memory == 0).float().mean().item()
    
    @staticmethod
    def _compute_memory_coherence(memory: torch.Tensor) -> float:
        # Compute cosine similarity between adjacent memory slots
        cos_sim = torch.nn.functional.cosine_similarity(
            memory[:-1], memory[1:], dim=-1
        )
        return cos_sim.mean().item()

def split_patterns_for_validation(
    patterns: Dict[str, torch.Tensor], 
    val_split: float
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    train_patterns = {}
    val_patterns = {}
    
    for pattern_type, pattern_data in patterns.items():
        split_idx = int(pattern_data.size(0) * (1 - val_split))
        train_patterns[pattern_type] = pattern_data[:split_idx]
        val_patterns[pattern_type] = pattern_data[split_idx:]
        
    return train_patterns, val_patterns