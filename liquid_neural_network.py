"""
Implementation of Liquid Neural Networks (LNNs) as described in the paper
"Liquid Time-constant Networks" by Hasani et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import math

class LiquidODECell(nn.Module):
    """
    Liquid Neural Network Cell implementing continuous-time neural dynamics
    through ordinary differential equations (ODEs).
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Learnable time constants (tau)
        self.tau = Parameter(torch.Tensor(hidden_dim))
        
        # Input-to-hidden weights
        self.W_ih = Parameter(torch.Tensor(hidden_dim, input_dim))
        
        # Hidden-to-hidden weights
        self.W_hh = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        
        # Biases
        self.bias = Parameter(torch.Tensor(hidden_dim))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters with appropriate distributions"""
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
        # Initialize time constants with positive values
        self.tau.data.uniform_(0.1, 1.0)
    
    def forward(self, x, h, dt=0.1):
        """
        Forward pass implementing the ODE:
        dh/dt = -h/tau + σ(W_ih·x + W_hh·h + b)
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            h: Hidden state tensor of shape (batch_size, hidden_dim)
            dt: Time step for Euler integration
            
        Returns:
            Updated hidden state after one time step
        """
        # Compute activation
        activation = F.tanh(F.linear(x, self.W_ih) + F.linear(h, self.W_hh) + self.bias)
        
        # ODE update using Euler method
        dh = (-h / self.tau.unsqueeze(0) + activation) * dt
        
        # Update hidden state
        new_h = h + dh
        
        return new_h


class LiquidNeuralNetwork(nn.Module):
    """
    Liquid Neural Network model implementing continuous-time neural dynamics
    for sequence processing.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dt=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dt = dt
        
        # Create LNN cells for each layer
        self.cells = nn.ModuleList([
            LiquidODECell(
                input_dim if i == 0 else hidden_dim,
                hidden_dim
            ) for i in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, h=None):
        """
        Forward pass through the LNN for a sequence of inputs.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            h: Optional initial hidden state. If None, initialized with zeros.
            
        Returns:
            outputs: Output tensor of shape (batch_size, seq_len, output_dim)
            hidden: Final hidden state
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state if not provided
        if h is None:
            h = [torch.zeros(batch_size, self.hidden_dim, device=x.device) 
                 for _ in range(self.num_layers)]
        
        outputs = []
        
        # Process each time step
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Process through each layer
            for i, cell in enumerate(self.cells):
                h[i] = cell(x_t if i == 0 else h[i-1], h[i], self.dt)
            
            # Compute output for this time step
            out_t = self.output_proj(h[-1])
            outputs.append(out_t)
        
        # Stack outputs along sequence dimension
        outputs = torch.stack(outputs, dim=1)
        
        return outputs, h
    
    def predict(self, x):
        """
        Make predictions using the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Predictions tensor of shape (batch_size, seq_len, output_dim)
        """
        outputs, _ = self.forward(x)
        return outputs


class DynamicWeightLNN(LiquidNeuralNetwork):
    """
    Enhanced Liquid Neural Network with dynamic weight parameterization.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dt=0.1):
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dt)
        
        # Dynamic weight networks for each layer
        self.weight_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for time input
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim)
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x, h=None, context=None):
        """
        Forward pass with dynamic weight parameterization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            h: Optional initial hidden state
            context: Optional context vector for weight modulation
            
        Returns:
            outputs: Output tensor
            hidden: Final hidden state
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state if not provided
        if h is None:
            h = [torch.zeros(batch_size, self.hidden_dim, device=x.device) 
                 for _ in range(self.num_layers)]
        
        # Initialize context if not provided
        if context is None:
            context = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        outputs = []
        
        # Process each time step
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Normalized time for weight parameterization
            norm_time = torch.ones(batch_size, 1, device=x.device) * (t / seq_len)
            
            # Process through each layer with dynamic weights
            for i, (cell, weight_net) in enumerate(zip(self.cells, self.weight_networks)):
                # Compute dynamic weight parameters based on current state and time
                weight_input = torch.cat([h[i], norm_time], dim=1)
                weight_params = weight_net(weight_input)
                
                # Reshape to weight matrix
                dynamic_weights = weight_params.view(batch_size, self.hidden_dim, self.hidden_dim)
                
                # Apply dynamic weights in the ODE update
                h[i] = self._dynamic_update(x_t if i == 0 else h[i-1], h[i], 
                                          cell, dynamic_weights, context)
            
            # Compute output for this time step
            out_t = self.output_proj(h[-1])
            outputs.append(out_t)
        
        # Stack outputs along sequence dimension
        outputs = torch.stack(outputs, dim=1)
        
        return outputs, h
    
    def _dynamic_update(self, x, h, cell, dynamic_weights, context, dt=None):
        """
        Update hidden state with dynamic weights.
        
        Args:
            x: Input tensor
            h: Hidden state tensor
            cell: LiquidODECell
            dynamic_weights: Batch of dynamic weight matrices
            context: Context vector for weight modulation
            dt: Time step (if None, use default)
            
        Returns:
            Updated hidden state
        """
        if dt is None:
            dt = self.dt
        
        batch_size = h.shape[0]
        
        # Compute standard activation
        activation = F.tanh(F.linear(x, cell.W_ih) + F.linear(h, cell.W_hh) + cell.bias)
        
        # Apply dynamic weights (batch matrix multiplication)
        dynamic_activation = torch.bmm(dynamic_weights, h.unsqueeze(2)).squeeze(2)
        
        # Combine standard and dynamic activations with context modulation
        combined_activation = activation + dynamic_activation * torch.sigmoid(context)
        
        # ODE update using Euler method
        dh = (-h / cell.tau.unsqueeze(0) + combined_activation) * dt
        
        # Update hidden state
        new_h = h + dh
        
        return new_h
