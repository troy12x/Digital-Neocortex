"""
Space-Time Layers for the Digital Neocortex architecture.
These layers combine spatial decomposition and temporal evolution into unified systems,
inspired by the biological neocortex and Liquid Foundation Models (LFMs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange, repeat
from torchdiffeq import odeint, odeint_adjoint

from ode_solvers import create_solver


class LiquidTimeLayer(nn.Module):
    """
    Implements continuous-time neural dynamics based on Liquid Neural Networks.
    Each neuron evolves according to a nonlinear ODE.
    """
    def __init__(self, dim, hidden_dim=None, activation='tanh', learn_tau=True, 
                 solver='dopri5', integration_time=1.0, adjoint=False):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim
        self.integration_time = integration_time
        self.solver = solver
        self.adjoint = adjoint
        
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
    
    def dynamics(self, t, x, u=None):
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
    
    def forward(self, x, u=None, integration_time=None):
        """
        Evolve the state through time using ODE integration.
        
        Args:
            x: Initial state tensor [batch_size, dim]
            u: Optional input tensor [batch_size, dim]
            integration_time: Optional override for integration time
            
        Returns:
            Evolved state tensor [batch_size, dim]
        """
        integration_time = integration_time or self.integration_time
        
        # Integration time points
        t = torch.tensor([0, integration_time], device=x.device, dtype=torch.float32)
        
        # Select ODE solver
        if self.adjoint:
            solver_func = odeint_adjoint
        else:
            solver_func = odeint
        
        # Solve ODE
        solution = solver_func(
            self.dynamics,
            x,
            t,
            method=self.solver,
            args=(u,)
        )
        
        # Return final state
        return solution[-1]


class SpatialDecompositionLayer(nn.Module):
    """
    Decomposes input into spatial units and processes each independently.
    """
    def __init__(self, dim, num_spatial_units=8, hidden_dim=None, activation='tanh'):
        super().__init__()
        self.dim = dim
        self.num_spatial_units = num_spatial_units
        self.unit_dim = dim // num_spatial_units
        self.hidden_dim = hidden_dim or self.unit_dim
        
        # Ensure dimension is divisible by number of spatial units
        assert dim % num_spatial_units == 0, f"Dimension {dim} must be divisible by {num_spatial_units}"
        
        # Spatial unit processors
        self.spatial_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.unit_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                getattr(nn, activation.upper())() if hasattr(nn, activation.upper()) else nn.Tanh(),
                nn.Linear(self.hidden_dim, self.unit_dim)
            )
            for _ in range(num_spatial_units)
        ])
        
        # Spatial attention for coupling
        self.spatial_attention = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        """
        Process input through spatial decomposition.
        
        Args:
            x: Input tensor [batch_size, dim]
            
        Returns:
            Processed tensor [batch_size, dim]
        """
        batch_size = x.shape[0]
        
        # Split input into spatial units
        x_split = torch.split(x, self.unit_dim, dim=1)
        
        # Process each spatial unit
        processed_units = []
        for i, unit in enumerate(self.spatial_processors):
            processed_units.append(unit(x_split[i]))
        
        # Concatenate processed units
        x_processed = torch.cat(processed_units, dim=1)
        
        # Apply spatial attention
        attention = self.spatial_attention(x)
        x_attended = x_processed * attention
        
        return x_attended


class TemporalFusionLayer(nn.Module):
    """
    Fuses information across time using attention-like mechanisms.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Ensure dimension is divisible by number of heads
        assert dim % num_heads == 0, f"Dimension {dim} must be divisible by {num_heads}"
        
        # Multi-head projection
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
        # Output projection
        self.output_proj = nn.Linear(dim, dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, context=None):
        """
        Fuse information across time.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim] or [batch_size, dim]
            context: Optional context tensor with same shape as x
            
        Returns:
            Fused tensor with same shape as x
        """
        # Handle non-sequence input
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            if context is not None:
                context = context.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        
        # Use input as context if not provided
        if context is None:
            context = x
        
        # Compute Q, K, V
        q = self.query(x)
        k = self.key(context)
        v = self.value(context)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply softmax
        attn = F.softmax(attn, dim=-1)
        
        # Apply dropout
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Apply output projection
        out = self.output_proj(out)
        
        # Squeeze if input was non-sequence
        if seq_len == 1:
            out = out.squeeze(1)
        
        return out


class SpaceTimeLayer(nn.Module):
    """
    Combines spatial decomposition and temporal evolution into a unified system.
    """
    def __init__(self, dim, num_spatial_units=8, hidden_dim=None, activation='tanh',
                 integration_time=1.0, solver='dopri5', adjoint=False, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_spatial_units = num_spatial_units
        self.hidden_dim = hidden_dim or dim
        self.integration_time = integration_time
        
        # Spatial decomposition
        self.spatial_decomp = SpatialDecompositionLayer(
            dim, num_spatial_units, hidden_dim, activation
        )
        
        # Liquid time layers for each spatial unit
        self.unit_dim = dim // num_spatial_units
        self.liquid_layers = nn.ModuleList([
            LiquidTimeLayer(
                self.unit_dim, hidden_dim // num_spatial_units, activation,
                learn_tau=True, solver=solver, integration_time=integration_time,
                adjoint=adjoint
            )
            for _ in range(num_spatial_units)
        ])
        
        # Temporal fusion
        self.temporal_fusion = TemporalFusionLayer(dim, num_heads=8, dropout=dropout)
        
        # Coupling weights for spatial units
        self.coupling_weights = nn.Parameter(torch.ones(num_spatial_units) / num_spatial_units)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
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
        
        # Apply spatial decomposition
        x_spatial = self.spatial_decomp(x)
        x_spatial = self.norm1(x_spatial)
        
        # Split into spatial units
        x_split = torch.split(x_spatial, self.unit_dim, dim=1)
        u_split = None
        if u is not None:
            u_split = torch.split(u, self.unit_dim, dim=1)
        
        # Process each spatial unit through liquid time layers
        outputs = []
        for i, liquid_layer in enumerate(self.liquid_layers):
            # Get input for this unit
            x_i = x_split[i]
            
            # Get external input for this unit (if provided)
            u_i = None
            if u_split is not None:
                u_i = u_split[i]
            
            # Process through liquid layer
            out_i = liquid_layer(x_i, u_i, integration_time)
            outputs.append(out_i)
        
        # Concatenate outputs from all spatial units
        x_liquid = torch.cat(outputs, dim=1)
        
        # Apply temporal fusion
        x_fused = self.temporal_fusion(x_liquid)
        x_fused = self.dropout(x_fused)
        
        # Apply coupling weights
        coupling = F.softmax(self.coupling_weights, dim=0)
        x_split = torch.split(x_fused, self.unit_dim, dim=1)
        x_coupled = torch.zeros_like(x_fused)
        
        for i, x_i in enumerate(x_split):
            x_coupled[:, i*self.unit_dim:(i+1)*self.unit_dim] = coupling[i] * x_i
        
        # Residual connection and normalization
        x_out = x + x_coupled
        x_out = self.norm2(x_out)
        
        return x_out


class S5Layer(nn.Module):
    """
    Simplified State Space Layer (S5) for efficient sequence modeling.
    Based on the principles of linear state space models with diagonal state matrices.
    """
    def __init__(self, d_model, d_state=64, dropout=0.0, init_scale=1.0, seq_len=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.seq_len = seq_len
        
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
        
        # Pre-compute discretized matrices for fixed sequence length if provided
        if seq_len is not None:
            self.register_buffer('cached_K', None)  # Will be computed in first forward pass
    
    def discretize(self):
        """
        Discretize the continuous-time system using ZOH method.
        
        Returns:
            Lambda_bar: Discretized state matrix
            B_bar: Discretized input matrix
        """
        # Create complex diagonal state matrix
        Lambda = torch.complex(self.Lambda_re, self.Lambda_im)
        
        # Discretize the system (ZOH discretization)
        Lambda_bar = torch.exp(Lambda * self.Delta)
        B_bar = (Lambda_bar - 1.0) / Lambda.unsqueeze(1) * self.B
        
        return Lambda_bar, B_bar
    
    def compute_kernel(self, seq_len):
        """
        Compute the convolution kernel for the SSM.
        
        Args:
            seq_len: Length of the sequence
            
        Returns:
            K: Convolution kernel [seq_len, d_model, d_model]
        """
        Lambda_bar, B_bar = self.discretize()
        
        # Compute powers of Lambda_bar for each time step
        Lambda_powers = torch.zeros(seq_len, self.d_state, dtype=torch.complex64, device=self.Lambda_re.device)
        Lambda_powers[0] = torch.ones_like(Lambda_bar)
        for t in range(1, seq_len):
            Lambda_powers[t] = Lambda_powers[t-1] * Lambda_bar
        
        # Compute kernel
        K = torch.einsum('t...,ij,jk->tik', Lambda_powers, B_bar, self.C)
        
        return K.real
    
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
        
        # Check if we need to compute or use cached kernel
        if self.seq_len is not None and seq_len <= self.seq_len and self.cached_K is not None:
            # Use cached kernel
            K = self.cached_K[:seq_len]
        else:
            # Compute kernel
            K = self.compute_kernel(seq_len)
            
            # Cache kernel if sequence length is fixed
            if self.seq_len is not None and seq_len == self.seq_len:
                self.cached_K = K
        
        # Apply convolution
        u_flat = u.reshape(batch_size * seq_len, -1)
        y_flat = torch.matmul(u_flat, K.transpose(1, 2).reshape(seq_len * self.d_model, self.d_model))
        y = y_flat.reshape(batch_size, seq_len, self.d_model)
        
        # Add direct feedthrough
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)
        
        # Apply dropout
        y = self.dropout(y)
        
        # Compute final state for recurrent use
        if state is None:
            state = torch.zeros(batch_size, self.d_state, dtype=torch.complex64, device=u.device)
        
        Lambda_bar, B_bar = self.discretize()
        for t in range(seq_len):
            state = Lambda_bar.unsqueeze(0) * state + torch.matmul(u[:, t, :], B_bar.t())
        
        return y, state.real


class SpaceTimeSequenceLayer(nn.Module):
    """
    Space-Time layer for processing sequences, combining spatial decomposition,
    temporal evolution, and efficient sequence modeling.
    """
    def __init__(self, dim, seq_len=None, num_spatial_units=8, hidden_dim=None, 
                 activation='tanh', integration_time=1.0, use_s5=True, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.use_s5 = use_s5
        
        # Space-Time processing
        self.space_time = SpaceTimeLayer(
            dim, num_spatial_units, hidden_dim, activation,
            integration_time, dropout=dropout
        )
        
        # S5 layer for sequence modeling (if enabled)
        if use_s5:
            self.s5 = S5Layer(dim, d_state=dim//2, dropout=dropout, seq_len=seq_len)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        if use_s5:
            self.norm2 = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, state=None):
        """
        Process a sequence through the space-time sequence layer.
        
        Args:
            x: Input sequence [batch_size, seq_len, dim]
            state: Optional initial state for S5 layer
            
        Returns:
            Output sequence after processing [batch_size, seq_len, dim]
            state_final: Final state (if S5 is used)
        """
        batch_size, seq_len, _ = x.shape
        
        # Process each time step independently through space-time layer
        outputs = []
        for t in range(seq_len):
            out_t = self.space_time(x[:, t, :])
            outputs.append(out_t)
        
        # Stack outputs
        x_st = torch.stack(outputs, dim=1)
        x_st = self.norm1(x_st)
        x_st = self.dropout(x_st)
        
        # Apply S5 layer for sequence modeling (if enabled)
        if self.use_s5:
            x_s5, state_final = self.s5(x_st, state)
            x_out = self.norm2(x_st + x_s5)
            return x_out, state_final
        else:
            return x_st, None
