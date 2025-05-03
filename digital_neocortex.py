"""
Digital Neocortex: A neural network architecture inspired by the biological neocortex,
incorporating Liquid Foundation Models (LFMs) and State Space Models (S5) principles.

This implementation focuses on continuous-time neural dynamics, space-time processing,
and hierarchical abstraction similar to the human neocortex.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange, repeat
from torchdiffeq import odeint, odeint_adjoint
import sympy as sp


class ContinuousTimeNeuron(nn.Module):
    """
    Implements a continuous-time neuron with ODE dynamics similar to Liquid Neural Networks.
    Each neuron evolves according to: dx_i(t)/dt = -x_i(t)/τ_i + σ(∑_j W_ij(t)x_j(t) + u_i(t))
    """
    def __init__(self, dim, hidden_dim=None, activation='tanh', learn_tau=True):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim
        
        # Learnable time constants
        if learn_tau:
            self.tau = nn.Parameter(torch.ones(dim) * 10.0)  # Initialize with reasonable time scale
        else:
            self.register_buffer('tau', torch.ones(dim) * 10.0)
        
        # Weight networks (dynamic parameterization)
        self.weight_net = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(dim, dim)
        
        # Activation function
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, t, x, u=None):
        """
        Compute the derivative dx/dt for the ODE solver.
        
        Args:
            t: Time (scalar)
            x: State tensor [batch_size, dim]
            u: Optional input tensor [batch_size, dim]
            
        Returns:
            dx/dt: State derivative [batch_size, dim]
        """
        # Decay term
        decay = -x / self.tau
        
        # Dynamic weight computation
        weights = self.weight_net(x)
        
        # Input term
        if u is not None:
            input_term = self.input_proj(u)
        else:
            input_term = 0
        
        # Compute derivative
        dx = decay + self.activation(weights + input_term)
        
        return dx


class SpaceTimeLayer(nn.Module):
    """
    Implements a space-time processing layer that combines spatial decomposition
    and temporal evolution into a unified system.
    """
    def __init__(self, dim, num_spatial_units=8, hidden_dim=None, activation='tanh', 
                 integration_time=1.0, solver='dopri5', adjoint=False):
        super().__init__()
        self.dim = dim
        self.num_spatial_units = num_spatial_units
        self.hidden_dim = hidden_dim or dim
        self.integration_time = integration_time
        self.solver = solver
        self.adjoint = adjoint
        
        # Create spatial units (each is a continuous-time neuron)
        self.spatial_units = nn.ModuleList([
            ContinuousTimeNeuron(dim // num_spatial_units, hidden_dim, activation)
            for _ in range(num_spatial_units)
        ])
        
        # Temporal fusion module (attention-like)
        self.temporal_fusion = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Tanh()
        )
        
        # Coupling weights for spatial units
        self.coupling_weights = nn.Parameter(torch.ones(num_spatial_units) / num_spatial_units)
    
    def forward(self, x, u=None, integration_time=None):
        """
        Process input through the space-time layer.
        
        Args:
            x: Input tensor [batch_size, dim]
            u: Optional external input [batch_size, dim]
            integration_time: Optional override for integration time
            
        Returns:
            Output tensor after space-time processing [batch_size, dim]
        """
        batch_size = x.shape[0]
        integration_time = integration_time or self.integration_time
        
        # Split input into spatial units
        x_split = torch.split(x, self.dim // self.num_spatial_units, dim=-1)
        u_split = None
        if u is not None:
            u_split = torch.split(u, self.dim // self.num_spatial_units, dim=-1)
        
        # Process each spatial unit
        outputs = []
        for i, unit in enumerate(self.spatial_units):
            # Initial state for this unit
            x_i = x_split[i]
            
            # External input for this unit (if provided)
            u_i = None
            if u_split is not None:
                u_i = u_split[i]
            
            # Solve ODE for this unit
            if self.adjoint:
                solver = odeint_adjoint
            else:
                solver = odeint
            
            # Integration time points
            t = torch.linspace(0, integration_time, 2, device=x.device)
            
            # Solve ODE
            solution = solver(
                unit,
                x_i,
                t,
                method=self.solver,
                args=(u_i,)
            )
            
            # Take final state
            outputs.append(solution[-1])
        
        # Concatenate outputs from all spatial units
        x_spatial = torch.cat(outputs, dim=-1)
        
        # Apply temporal fusion
        x_fused = self.temporal_fusion(x_spatial)
        
        # Apply coupling weights
        coupling = F.softmax(self.coupling_weights, dim=0)
        x_split = torch.split(x_fused, self.dim // self.num_spatial_units, dim=-1)
        x_coupled = torch.zeros_like(x_fused)
        
        for i, x_i in enumerate(x_split):
            x_coupled += coupling[i] * x_i
        
        return x_coupled


class StateSpaceLayer(nn.Module):
    """
    Implements a simplified State Space Layer (S5) for efficient sequence modeling.
    Based on the principles of linear state space models with diagonal state matrices.
    """
    def __init__(self, d_model, d_state=64, dropout=0.0, init_scale=1.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Initialize diagonal state matrix (Λ) using HiPPO-N approximation
        # This is a simplified version; a full implementation would use the actual HiPPO-N matrix
        self.Lambda_re = nn.Parameter(torch.randn(d_state) * 0.01 - 0.5)
        self.Lambda_im = nn.Parameter(torch.randn(d_state) * 0.01)
        
        # Input projection (B matrix)
        self.B = nn.Parameter(torch.randn(d_state, d_model) * init_scale / math.sqrt(d_state))
        
        # Output projection (C matrix)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * init_scale / math.sqrt(d_state))
        
        # Direct feedthrough (D matrix)
        self.D = nn.Parameter(torch.zeros(d_model))
        
        # Step size parameter (Δ)
        self.Delta = nn.Parameter(torch.ones(1) * 0.1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, u, state=None):
        """
        Process a batch of sequences through the SSM.
        
        Args:
            u: Input sequence [batch_size, seq_len, d_model]
            state: Optional initial state [batch_size, d_state]
            
        Returns:
            y: Output sequence [batch_size, seq_len, d_model]
            state_final: Final state [batch_size, d_state]
        """
        batch_size, seq_len, _ = u.shape
        
        # Create complex diagonal state matrix
        Lambda = torch.complex(self.Lambda_re, self.Lambda_im)
        
        # Discretize the system (ZOH discretization)
        Lambda_bar = torch.exp(Lambda * self.Delta)
        B_bar = (Lambda_bar - 1.0) / Lambda.unsqueeze(1) * self.B
        
        # Initialize state if not provided
        if state is None:
            state = torch.zeros(batch_size, self.d_state, dtype=torch.complex64, device=u.device)
        
        # Parallel scan implementation (simplified for clarity)
        # In practice, this would use a more efficient parallel scan algorithm
        outputs = []
        for t in range(seq_len):
            # Update state
            state = Lambda_bar.unsqueeze(0) * state + B_bar @ u[:, t, :].unsqueeze(-1)
            
            # Compute output
            y = (self.C @ state.unsqueeze(-1)).squeeze(-1) + self.D * u[:, t, :]
            outputs.append(y)
        
        # Stack outputs
        y = torch.stack(outputs, dim=1)
        
        # Apply dropout
        y = self.dropout(y)
        
        return y.real, state


class CorticalColumn(nn.Module):
    """
    Implements a modular processing unit mimicking cortical columns in the neocortex.
    Each column specializes in processing specific types of patterns.
    """
    def __init__(self, dim, hidden_dim=None, activation='tanh', integration_time=1.0):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim * 2
        
        # Continuous-time processing
        self.liquid_layer = ContinuousTimeNeuron(dim, hidden_dim, activation)
        
        # Normalization and processing
        self.norm = nn.LayerNorm(dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, dim)
        )
        
        # Integration time
        self.integration_time = integration_time
    
    def forward(self, x, u=None):
        """
        Process input through the cortical column.
        
        Args:
            x: Input tensor [batch_size, dim]
            u: Optional external input [batch_size, dim]
            
        Returns:
            Processed output [batch_size, dim]
        """
        # Solve ODE for continuous-time processing
        t = torch.linspace(0, self.integration_time, 2, device=x.device)
        solution = odeint(self.liquid_layer, x, t, args=(u,))
        
        # Take final state
        x_liquid = solution[-1]
        
        # Apply normalization
        x_norm = self.norm(x_liquid)
        
        # Apply feed-forward network
        x_ffn = self.ffn(x_norm)
        
        # Residual connection
        x_out = x_liquid + x_ffn
        
        return x_out


class MixtureOfExperts(nn.Module):
    """
    Implements a Mixture of Experts layer where each expert is a CorticalColumn.
    Only the top-k experts are activated for each input.
    """
    def __init__(self, dim, hidden_dim=None, num_experts=8, topk=2, activation='tanh', integration_time=1.0):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)
        
        # Create experts (cortical columns)
        self.experts = nn.ModuleList([
            CorticalColumn(dim, hidden_dim, activation, integration_time) 
            for _ in range(num_experts)
        ])
        
        # Router network to select experts
        self.router = nn.Linear(dim, num_experts)
    
    def forward(self, x, u=None):
        """
        Process input through the mixture of experts.
        
        Args:
            x: Input tensor [batch_size, dim]
            u: Optional external input [batch_size, dim]
            
        Returns:
            Processed output [batch_size, dim]
        """
        batch_size = x.shape[0]
        
        # Get routing probabilities
        routing_logits = self.router(x)
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        # Select top-k experts
        topk_probs, topk_indices = torch.topk(routing_probs, self.topk, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process with selected experts
        for i in range(self.topk):
            # For each position in the batch, select the appropriate expert
            for b in range(batch_size):
                expert_idx = topk_indices[b, i]
                expert_prob = topk_probs[b, i]
                expert_output = self.experts[expert_idx](x[b:b+1], u[b:b+1] if u is not None else None)
                output[b:b+1] += expert_prob * expert_output
        
        return output


class SparseAttention(nn.Module):
    """
    Implements a sparse attention mechanism that activates only a small subset of neurons,
    mimicking the sparsity principle of the neocortex.
    """
    def __init__(self, dim, topk_ratio=0.1, temperature=1.0):
        super().__init__()
        self.dim = dim
        self.topk_ratio = topk_ratio
        self.temperature = temperature
        
        # Projections
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
        # Scaling factor
        self.scale = dim ** -0.5
    
    def forward(self, x):
        """
        Apply sparse attention to the input.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            
        Returns:
            Attention output [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V projections
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Compute attention scores
        attn = torch.bmm(q, k.transpose(-2, -1)) * self.scale
        
        # Apply temperature
        attn = attn / self.temperature
        
        # Compute sparse attention mask (keep only top-k connections per query)
        topk = max(1, int(seq_len * self.topk_ratio))
        topk_values, topk_indices = torch.topk(attn, topk, dim=-1)
        
        # Create sparse attention mask
        mask = torch.zeros_like(attn)
        for b in range(batch_size):
            for i in range(seq_len):
                mask[b, i, topk_indices[b, i]] = 1
        
        # Apply mask and softmax
        attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(attn, v)
        
        return out


class HebbianLayer(nn.Module):
    """
    Implements Hebbian learning principle: "Neurons that fire together wire together"
    by dynamically adjusting connection strengths based on co-activation.
    """
    def __init__(self, dim, learning_rate=0.01):
        super().__init__()
        self.dim = dim
        self.learning_rate = learning_rate
        self.weight = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x, apply_hebbian=True):
        """
        Apply Hebbian learning to the input.
        
        Args:
            x: Input tensor [batch_size, dim]
            apply_hebbian: Whether to apply Hebbian learning
            
        Returns:
            Output tensor [batch_size, dim]
        """
        # Forward pass
        out = F.linear(x, self.weight, self.bias)
        
        # Apply Hebbian learning during training
        if apply_hebbian and self.training:
            # Compute co-activation matrix
            batch_size = x.shape[0]
            co_activation = torch.bmm(x.unsqueeze(2), x.unsqueeze(1))
            co_activation = co_activation.mean(dim=0)
            
            # Update weights based on Hebbian rule
            with torch.no_grad():
                self.weight.data += self.learning_rate * co_activation
                
                # Normalize weights to prevent explosion
                norm = torch.norm(self.weight.data, dim=1, keepdim=True)
                self.weight.data = self.weight.data / (norm + 1e-6)
        
        return out


class HierarchicalLayer(nn.Module):
    """
    Processes information in hierarchical layers, from raw input to abstract representation,
    similar to the hierarchical processing in the neocortex.
    """
    def __init__(self, dim, hidden_dim=None, num_experts=8, expert_topk=2, 
                 sequence_length=None, activation='tanh', integration_time=1.0):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim * 2
        self.sequence_length = sequence_length
        
        # Layer 1: Sparse attention for pattern recognition (if sequence input)
        if sequence_length is not None:
            self.sparse_attn = SparseAttention(dim)
            self.norm1 = nn.LayerNorm(dim)
        
        # Layer 2: Space-time processing
        self.space_time = SpaceTimeLayer(dim, num_spatial_units=8, hidden_dim=hidden_dim, 
                                         activation=activation, integration_time=integration_time)
        self.norm2 = nn.LayerNorm(dim)
        
        # Layer 3: Mixture of experts for specialized processing
        self.moe = MixtureOfExperts(dim, hidden_dim, num_experts, expert_topk, 
                                    activation=activation, integration_time=integration_time)
        self.norm3 = nn.LayerNorm(dim)
        
        # Layer 4: Hebbian learning
        self.hebbian = HebbianLayer(dim)
        self.norm4 = nn.LayerNorm(dim)
    
    def forward(self, x, u=None):
        """
        Process input through the hierarchical layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim] or [batch_size, dim]
            u: Optional external input
            
        Returns:
            Processed output with same shape as input
        """
        # Handle sequence input if provided
        if self.sequence_length is not None and len(x.shape) == 3:
            # Apply sparse attention
            x_attn = self.sparse_attn(x)
            x = x + x_attn
            x = self.norm1(x)
            
            # For sequence input, we process each time step independently
            batch_size, seq_len, _ = x.shape
            x_flat = x.reshape(-1, self.dim)
            
            # Apply space-time processing
            x_st = self.space_time(x_flat, u.reshape(-1, self.dim) if u is not None else None)
            x_flat = x_flat + x_st
            x_flat = self.norm2(x_flat)
            
            # Apply mixture of experts
            x_moe = self.moe(x_flat)
            x_flat = x_flat + x_moe
            x_flat = self.norm3(x_flat)
            
            # Apply Hebbian learning
            x_hebb = self.hebbian(x_flat)
            x_flat = x_flat + x_hebb
            x_flat = self.norm4(x_flat)
            
            # Reshape back to sequence
            x = x_flat.reshape(batch_size, seq_len, self.dim)
        else:
            # Apply space-time processing
            x_st = self.space_time(x, u)
            x = x + x_st
            x = self.norm2(x)
            
            # Apply mixture of experts
            x_moe = self.moe(x)
            x = x + x_moe
            x = self.norm3(x)
            
            # Apply Hebbian learning
            x_hebb = self.hebbian(x)
            x = x + x_hebb
            x = self.norm4(x)
        
        return x


class DigitalNeocortex(nn.Module):
    """
    Main Digital Neocortex architecture that combines all neocortex-inspired components
    with Liquid Foundation Models (LFMs) and State Space Models (S5) principles.
    """
    def __init__(
        self,
        vocab_size=None,
        dim=512,
        hidden_dim=1024,
        num_layers=6,
        num_experts=8,
        expert_topk=2,
        max_seq_len=1024,
        activation='tanh',
        integration_time=1.0,
        use_state_space=True,
        dropout=0.1
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.use_state_space = use_state_space
        
        # Token embedding for NLP tasks
        if vocab_size is not None:
            self.token_embedding = nn.Embedding(vocab_size, dim)
            self.output_proj = nn.Linear(dim, vocab_size)
        
        # Positional encoding (alternative to traditional positional embeddings)
        if max_seq_len is not None:
            # Use State Space Layer for positional encoding if enabled
            if use_state_space:
                self.pos_encoding = StateSpaceLayer(dim, d_state=dim//2, dropout=dropout)
            else:
                # Otherwise use traditional positional embeddings
                self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, dim))
                self._init_pos_embedding()
        
        # Hierarchical layers
        self.layers = nn.ModuleList([
            HierarchicalLayer(
                dim=dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                expert_topk=expert_topk,
                sequence_length=max_seq_len if i == 0 else None,  # Only first layer handles sequences
                activation=activation,
                integration_time=integration_time
            )
            for i in range(num_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _init_pos_embedding(self):
        """Initialize positional embeddings with sine and cosine functions."""
        if not hasattr(self, 'pos_embedding'):
            return
            
        position = torch.arange(self.max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2) * (-math.log(10000.0) / self.dim))
        pos_emb = torch.zeros(1, self.max_seq_len, self.dim)
        pos_emb[0, :, 0::2] = torch.sin(position * div_term)
        pos_emb[0, :, 1::2] = torch.cos(position * div_term)
        self.pos_embedding.data = pos_emb
    
    def forward(self, x, return_embeddings=False):
        """
        Process input through the Digital Neocortex.
        
        Args:
            x: Input tensor, either token indices [batch_size, seq_len] or 
               features [batch_size, seq_len, dim]
            return_embeddings: Whether to return embeddings instead of logits
            
        Returns:
            Output tensor with appropriate shape for the task
        """
        # Handle token indices input (NLP tasks)
        if hasattr(self, 'token_embedding') and x.dtype == torch.long:
            x = self.token_embedding(x)
        
        # Apply positional encoding/embedding
        if hasattr(self, 'pos_encoding') and len(x.shape) == 3:
            # Use State Space Layer for positional encoding
            x, _ = self.pos_encoding(x)
        elif hasattr(self, 'pos_embedding') and len(x.shape) == 3:
            # Add positional embeddings
            seq_len = x.shape[1]
            x = x + self.pos_embedding[:, :seq_len, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Process through hierarchical layers
        for layer in self.layers:
            x = layer(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Return embeddings if requested
        if return_embeddings:
            return x
        
        # Project to vocabulary (for NLP tasks)
        if hasattr(self, 'output_proj'):
            x = self.output_proj(x)
        
        return x
    
    def generate(self, prompt, max_length=100, temperature=1.0, top_k=50):
        """
        Generate text from a prompt (for NLP tasks).
        
        Args:
            prompt: Input tensor of token indices [batch_size, seq_len]
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Number of top tokens to sample from
            
        Returns:
            Generated sequence [batch_size, seq_len + generated_len]
        """
        if not hasattr(self, 'token_embedding') or not hasattr(self, 'output_proj'):
            raise ValueError("Model not configured for text generation")
        
        self.eval()
        
        # Generate tokens
        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions
                logits = self(prompt)
                
                # Focus on the last token's prediction
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                
                # Sample from the filtered distribution
                probs = F.softmax(top_k_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Map back to vocabulary indices
                next_token = torch.gather(top_k_indices, -1, next_token)
                
                # Append to the sequence
                prompt = torch.cat([prompt, next_token], dim=-1)
                
                # Break if we exceed max sequence length
                if prompt.size(1) >= self.max_seq_len:
                    break
        
        return prompt
