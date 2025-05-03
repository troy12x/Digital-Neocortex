"""
Implementation of Digital Neocortex components that enhance Liquid Neural Networks.
This includes:
1. Cortical Column Organization (Mixture of Experts)
2. Sparse Activation Mechanism
3. Space-Time Processing Layers
4. Enhanced Hebbian Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np


class CorticalColumn(nn.Module):
    """
    Implementation of a cortical column as a specialized expert in a Mixture of Experts approach.
    Each expert specializes in processing specific types of patterns.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Expert network
        self.expert = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """Process input through the expert"""
        return self.expert(x)


class CorticalColumnOrganization(nn.Module):
    """
    Implementation of the Ω function that channels dynamic representations into
    specialized processing pathways (cortical columns).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=8, top_k=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Create experts (cortical columns)
        self.experts = nn.ModuleList([
            CorticalColumn(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_experts)
        )
    
    def forward(self, x, t=None, return_gates=False):
        """
        Forward pass implementing the Ω function.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            t: Optional time tensor for time-dependent routing
            return_gates: Whether to return gating values
            
        Returns:
            Output tensor after routing through experts
        """
        batch_size = x.shape[0]
        
        # Compute routing scores
        routing_scores = self.router(x)
        
        # Create sparsity mask for top-k experts
        _, indices = torch.topk(routing_scores, self.top_k, dim=1)
        mask = torch.zeros_like(routing_scores).scatter_(1, indices, 1)
        
        # Apply softmax for normalized gating
        gates = F.softmax(routing_scores * mask, dim=1)
        
        # Process input through each expert and combine with gates
        expert_outputs = torch.zeros(batch_size, self.output_dim, device=x.device)
        
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            expert_outputs += expert_out * gates[:, i].unsqueeze(1)
        
        if return_gates:
            return expert_outputs, gates
        
        return expert_outputs


class SparseActivation(nn.Module):
    """
    Implementation of the Ψ function that focuses computational resources on the
    most relevant information through sparse activation mechanisms.
    """
    def __init__(self, input_dim, hidden_dim, sparsity=0.1, lateral_inhibition=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity = sparsity  # Target sparsity level (ρ)
        self.lateral_inhibition = lateral_inhibition  # λ parameter
        
        # Learnable parameters
        self.time_constant = Parameter(torch.Tensor(hidden_dim))
        self.connectivity = Parameter(torch.Tensor(hidden_dim, input_dim))
        self.threshold = Parameter(torch.Tensor(1))
        self.learning_rate = Parameter(torch.Tensor(1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        stdv = 0.1 / math.sqrt(self.hidden_dim)
        self.connectivity.data.uniform_(-stdv, stdv)
        self.threshold.data.fill_(0.1)
        self.learning_rate.data.fill_(0.01)
        
        # Initialize time constants with positive values (ensure they're not too small)
        self.time_constant.data.uniform_(0.5, 1.0)
    
    def forward(self, z, prev_a=None, prev_threshold=None, dt=0.1, context=None):
        """
        Forward pass implementing the Ψ function.
        
        Args:
            z: Input tensor from cortical column organization
            prev_a: Previous activation levels (if None, initialized with zeros)
            prev_threshold: Previous threshold value
            dt: Time step for integration
            context: Optional context vector
            
        Returns:
            Sparse activation pattern
        """
        batch_size = z.shape[0]
        
        # Initialize activations if not provided
        if prev_a is None:
            prev_a = torch.zeros(batch_size, self.hidden_dim, device=z.device)
        
        # Initialize threshold if not provided
        if prev_threshold is None:
            prev_threshold = self.threshold.expand(batch_size, 1)
        
        # Compute base connectivity
        S_base = self.connectivity
        
        # Apply context modulation if provided
        if context is not None:
            # Generate context-dependent connectivity modulation
            S_context = torch.matmul(context.unsqueeze(2), z.unsqueeze(1))
            S_context = S_context.view(batch_size, self.hidden_dim, self.input_dim)
            
            # Combine base and context-dependent connectivity
            S = S_base.unsqueeze(0) + 0.1 * S_context
        else:
            S = S_base.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Compute input drive
        input_drive = torch.bmm(S, z.unsqueeze(2)).squeeze(2)
        
        # Compute lateral inhibition
        inhibition = self.lateral_inhibition * torch.sum(prev_a, dim=1, keepdim=True)
        
        # Compute activation change (ODE) with numerical stability
        # Add a small epsilon to time constants to prevent division by zero
        time_const = self.time_constant.unsqueeze(0).clamp(min=0.1)
        
        # Compute each term separately with clipping to prevent extreme values
        decay_term = -prev_a / time_const
        decay_term = torch.clamp(decay_term, -10.0, 10.0)
        
        # Clip input drive and inhibition
        input_drive = torch.clamp(input_drive, -10.0, 10.0)
        inhibition = torch.clamp(inhibition, 0.0, 10.0)
        
        # Combine terms with a balanced dt for learning and stability
        da = (decay_term + input_drive - inhibition) * (dt * 0.3)
        
        # Update activations with clipping - allow slightly higher values
        a = torch.clamp(prev_a + da, 0.0, 15.0)
        
        # Compute average activation with clipping
        avg_activation = torch.mean(a, dim=1, keepdim=True)
        avg_activation = torch.clamp(avg_activation, 0.0, 1.0)
        
        # Update threshold to maintain target sparsity with improved adaptability
        dthreshold = self.learning_rate * (avg_activation - self.sparsity) * (dt * 0.3)
        dthreshold = torch.clamp(dthreshold, -0.2, 0.2)  # Allow more adaptation
        threshold = torch.clamp(prev_threshold + dthreshold, 0.01, 0.8)  # Lower max threshold
        
        # Apply threshold to get sparse activations
        sparse_a = torch.where(a > threshold, a, torch.zeros_like(a))
        
        # Final clipping to ensure no NaN values
        sparse_a = torch.clamp(sparse_a, 0.0, 10.0)
        
        return sparse_a, a, threshold


class SpaceTimeProcessing(nn.Module):
    """
    Implementation of the Φ function that structures the flow of information over time,
    creating a coherent sequence of processing steps.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_spatial_units=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_spatial_units = num_spatial_units
        
        # Spatial pathway
        self.spatial_units = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim // num_spatial_units)
            for _ in range(num_spatial_units)
        ])
        
        # Temporal pathway
        self.temporal_gating = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # State space model parameters
        self.A = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.B = Parameter(torch.Tensor(hidden_dim, input_dim))
        self.C = Parameter(torch.Tensor(output_dim, hidden_dim))
        self.D = Parameter(torch.Tensor(output_dim, input_dim))
        
        # Dynamic weighting function
        self.alpha_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        # Initialize state space model parameters
        nn.init.kaiming_uniform_(self.A)
        nn.init.kaiming_uniform_(self.B)
        nn.init.kaiming_uniform_(self.C)
        nn.init.kaiming_uniform_(self.D)
        
        # Make A stable by scaling its eigenvalues
        with torch.no_grad():
            # Compute eigenvalues
            eigvals = torch.linalg.eigvals(self.A)
            # Find maximum absolute eigenvalue
            max_eigval = torch.max(torch.abs(eigvals))
            # Scale A to have eigenvalues with magnitude < 1
            if max_eigval > 0.9:
                self.A.data *= 0.9 / max_eigval
    
    def forward(self, y, prev_h_t=None, prev_s=None, dt=0.1, context=None):
        """
        Forward pass implementing the Φ function.
        
        Args:
            y: Input tensor from sparse activation
            prev_h_t: Previous temporal hidden state
            prev_s: Previous state space model state
            dt: Time step for integration
            context: Optional context vector
            
        Returns:
            Processed output combining spatial and temporal pathways
        """
        batch_size = y.shape[0]
        
        # Process spatial pathway
        spatial_outputs = []
        for unit in self.spatial_units:
            spatial_outputs.append(unit(y))
        
        # Concatenate spatial outputs
        h_s = torch.cat(spatial_outputs, dim=1)
        
        # Initialize temporal state if not provided
        if prev_h_t is None:
            prev_h_t = torch.zeros(batch_size, self.hidden_dim, device=y.device)
        
        # Initialize state space model state if not provided
        if prev_s is None:
            prev_s = torch.zeros(batch_size, self.hidden_dim, device=y.device)
        
        # Compute gating parameter
        if context is not None:
            gamma_input = torch.cat([y, context], dim=1)
        else:
            gamma_input = torch.cat([y, prev_h_t], dim=1)
        
        gamma = self.temporal_gating(gamma_input)
        
        # Update state space model state
        ds = torch.matmul(prev_s, self.A.t()) + torch.matmul(y, self.B.t())
        s = prev_s + ds * dt
        
        # Compute state space model output
        g = torch.matmul(s, self.C.t()) + torch.matmul(y, self.D.t())
        
        # Update temporal hidden state
        h_t = gamma * prev_h_t + (1 - gamma) * g
        
        # Compute dynamic weighting
        if context is not None:
            alpha_input = torch.cat([y, context], dim=1)
        else:
            alpha_input = torch.cat([y, h_t], dim=1)
        
        alpha = self.alpha_net(alpha_input)
        
        # Combine spatial and temporal pathways
        h_st = alpha * h_s + (1 - alpha) * h_t
        
        return h_st, h_t, s


class EnhancedHebbianLearning(nn.Module):
    """
    Implementation of structured Hebbian learning that enhances the continuous
    weight adaptation already present in LNNs.
    """
    def __init__(self, input_dim, hidden_dim, num_experts=8, learning_rate=0.01, regularization=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.learning_rate = learning_rate
        self.regularization = regularization
        
        # Base weights
        self.base_weights = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        
        # Expert indicators (for pathway-specific adaptation)
        self.expert_indicators = Parameter(torch.Tensor(num_experts, hidden_dim), requires_grad=False)
        
        # Memory traces for each expert - stored as a Python list to avoid gradient tracking
        self.memory_traces = [torch.zeros(hidden_dim, hidden_dim) for _ in range(num_experts)]
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        stdv = 0.1 / math.sqrt(self.hidden_dim)  # Smaller initialization for stability
        self.base_weights.data.uniform_(-stdv, stdv)
        
        # Initialize expert indicators
        self.expert_indicators.data.zero_()
        neurons_per_expert = self.hidden_dim // self.num_experts
        
        for i in range(self.num_experts):
            start_idx = i * neurons_per_expert
            end_idx = start_idx + neurons_per_expert if i < self.num_experts - 1 else self.hidden_dim
            self.expert_indicators[i, start_idx:end_idx] = 1.0
    
    def forward(self, x, gates=None, dt=0.1):
        """
        Forward pass implementing structured Hebbian learning.
        
        Args:
            x: Input activation tensor
            gates: Gating values from cortical column organization
            dt: Time step for integration
            
        Returns:
            Updated weights and memory traces
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Compute co-activation patterns
        co_activation = torch.bmm(x.unsqueeze(2), x.unsqueeze(1))
        
        # Initialize weight updates - use a list to avoid in-place operations
        weight_update_terms = []
        
        # Create a new list for updated memory traces
        new_memory_traces = []
        
        # Update memory traces for each expert
        for i in range(self.num_experts):
            # Get expert indicator
            indicator = self.expert_indicators[i]
            
            # Compute mask for connections within this expert
            mask = torch.outer(indicator, indicator)
            
            # Get gating value for this expert
            if gates is not None:
                expert_gate = gates[:, i].mean()
            else:
                expert_gate = 1.0 / self.num_experts
            
            # Move memory trace to the correct device
            mem_trace = self.memory_traces[i].to(device)
            
            # Compute mean co-activation with numerical stability
            mean_co_activation = co_activation.mean(dim=0).detach()
            # Clip values to prevent NaN
            mean_co_activation = torch.clamp(mean_co_activation, -10.0, 10.0)
            
            # Update memory trace for this expert - avoid in-place operation
            # Use a small learning rate for stability
            new_memory = mem_trace + 0.01 * expert_gate * mean_co_activation * mask * dt
            new_memory_traces.append(new_memory.detach())  # Detach to avoid gradient tracking
            
            # Collect weight update terms instead of adding in-place
            # Use the original memory trace for the forward pass
            # Scale down the contribution to prevent NaN
            weight_update_terms.append(0.1 * expert_gate * mem_trace * mask)
        
        # Update memory traces after the forward pass
        self.memory_traces = new_memory_traces
        
        # Combine all weight update terms without in-place operations
        if weight_update_terms:
            weight_updates = sum(weight_update_terms)
        else:
            weight_updates = torch.zeros_like(self.base_weights)
        
        # Apply regularization without in-place operation - use a balanced regularization value
        weight_updates_with_reg = weight_updates - 0.05 * self.regularization * self.base_weights
        
        # Clip values to prevent NaN but allow more expressivity
        weight_updates_with_reg = torch.clamp(weight_updates_with_reg, -15.0, 15.0)
        
        # Compute final weights with an increased learning rate for better adaptation
        weights = self.base_weights + 0.05 * self.learning_rate * weight_updates_with_reg
        
        # Return weights and a copy of memory traces (not the actual list)
        return weights, new_memory_traces.copy()


class DigitalNeocortex(nn.Module):
    """
    Digital Neocortex: A biologically-inspired enhancement to Liquid Neural Networks.
    Combines cortical column organization, sparse activation, and temporal processing.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=4, top_k=2,
                 sparsity=0.3, dt=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dt = dt
        self.num_experts = num_experts
        self.sparsity = sparsity
        
        # For language tasks, use a more DynamicWeightLNN-like architecture
        # Core ODE function for continuous-time dynamics
        self.ode_func = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Dynamic weight network (similar to DynamicWeightLNN)
        self.weight_network = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        
        # Cortical Column Organization (simplified)
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_experts)
        )
        
        # Expert networks (cortical columns)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_experts)
        ])
        
        # Sparse activation
        self.sparse_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Hidden state
        self.hidden = None
    
    def forward(self, x, context=None):
        """
        Forward pass through the Digital Neocortex.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            context: Optional context vector
            
        Returns:
            Processed output tensor
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state if needed
        if self.hidden is None or self.hidden.shape[0] != batch_size:
            self.hidden = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        outputs = []
        h_t = self.hidden.detach()  # Start with detached hidden state
        
        # Process each time step
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Cortical Column Organization (Mixture of Experts)
            # Compute routing probabilities
            routing_logits = self.router(x_t)
            routing_probs = F.softmax(routing_logits, dim=1)
            
            # Apply experts
            expert_outputs = []
            for i, expert in enumerate(self.experts):
                expert_out = expert(x_t)
                # Weight by routing probability
                expert_outputs.append(expert_out * routing_probs[:, i].unsqueeze(1))
            
            # Combine expert outputs
            combined_experts = torch.stack(expert_outputs, dim=0).sum(dim=0)
            
            # Apply sparse activation
            sparsity_gate = self.sparse_gate(combined_experts)
            # Ensure sparsity by keeping only top k% of activations
            k = int(self.hidden_dim * self.sparsity)
            values, _ = torch.topk(sparsity_gate, k, dim=1)
            threshold = values[:, -1].unsqueeze(1)
            sparse_output = combined_experts * (sparsity_gate >= threshold).float()
            
            # Generate dynamic weights based on input and hidden state
            weight_input = torch.cat([x_t, h_t], dim=1)
            dynamic_weights = self.weight_network(weight_input)
            
            # Apply gradient clipping to prevent exploding gradients
            dynamic_weights = torch.clamp(dynamic_weights, -10.0, 10.0)
            
            dynamic_weights = dynamic_weights.view(batch_size, self.hidden_dim, self.hidden_dim)
            
            # Apply dynamic weights to sparse output
            weighted_output = torch.bmm(sparse_output.unsqueeze(1), dynamic_weights).squeeze(1)
            
            # Apply gradient clipping to prevent exploding gradients
            weighted_output = torch.clamp(weighted_output, -10.0, 10.0)
            
            # Compute ODE function
            f_h = self.ode_func(h_t)
            
            # Apply gradient clipping to prevent exploding gradients
            f_h = torch.clamp(f_h, -10.0, 10.0)
            
            # Update hidden state using Euler integration (similar to DynamicWeightLNN)
            # Use a smaller dt for more stable integration
            effective_dt = min(self.dt, 0.05)
            h_t = h_t + effective_dt * f_h + weighted_output * effective_dt
            
            # Apply gradient clipping to prevent exploding gradients
            h_t = torch.clamp(h_t, -10.0, 10.0)
            
            # Project to output dimension
            output = self.output_proj(h_t)
            
            outputs.append(output)
            
        # Update the stored hidden state
        self.hidden = h_t.detach()
        
        # Stack outputs along sequence dimension
        outputs = torch.stack(outputs, dim=1)
        
        return outputs
    
    def reset_state(self):
        """Reset internal states"""
        self.hidden = None
