import torch
import matplotlib.pyplot as plt
import seaborn as sns
from app import NeocortexConfig, DigitalNeocortex
from typing import Dict, List, Tuple
import numpy as np

class NeocortexMonitor:
    def __init__(self, neocortex: DigitalNeocortex):
        self.neocortex = neocortex
        self.attention_maps = []
        self.memory_states = []
        self.layer_activations = []
        self.prediction_errors = []
        
    def record_state(self, layer_output: torch.Tensor, attention_weights: torch.Tensor, 
                    memory_state: torch.Tensor, prediction_error: float):
        # Reshape tensors for visualization
        if len(layer_output.shape) == 3:
            layer_output = layer_output[:, 0]  # Take first timestep
        if len(attention_weights.shape) == 3:
            attention_weights = attention_weights[:, 0]
            
        # Convert to numpy for visualization
        self.layer_activations.append(layer_output.detach().cpu().numpy())
        self.attention_maps.append(attention_weights.detach().cpu().numpy())
        self.memory_states.append(memory_state.detach().cpu().numpy())
        self.prediction_errors.append(prediction_error)
        
    def visualize_processing(self):
        if not self.attention_maps or not self.layer_activations or not self.memory_states:
            print("No data to visualize yet")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot attention maps
        sns.heatmap(self.attention_maps[0], ax=axes[0,0])
        axes[0,0].set_title('Attention Map')
        
        # Plot memory states
        sns.heatmap(self.memory_states[0], ax=axes[0,1])
        axes[0,1].set_title('Memory State')
        
        # Plot layer activations
        sns.heatmap(self.layer_activations[0], ax=axes[1,0])
        axes[1,0].set_title('Layer Activations')
        
        # Plot prediction errors
        if self.prediction_errors:
            axes[1,1].plot(self.prediction_errors)
            axes[1,1].set_title('Prediction Errors')
        
        plt.tight_layout()
        plt.show()

def generate_complex_inputs(config: NeocortexConfig, batch_size: int = 2):
    """Generate more complex test inputs"""
    time_steps = 16  # Smaller sequence length
    t = torch.linspace(0, 4*np.pi, time_steps)
    sine_wave = torch.sin(t).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
    
    # Ensure all inputs match the input dimension
    sequence = torch.randn(batch_size, time_steps, config.input_dim)
    patterns = torch.zeros(batch_size, time_steps, config.input_dim)
    
    # Add temporal patterns
    sequence = sequence + sine_wave.expand(-1, -1, config.input_dim)
    patterns[:, ::4] = 1.0  # Add periodic patterns
    
    return {
        'text': sequence,
        'sensory': patterns
    }

def test_memory_system(neocortex: DigitalNeocortex, monitor: NeocortexMonitor):
    """Test memory retention and recall"""
    print("\nTesting Memory System...")
    
    # Test sequence
    sequence = torch.randn(1, 5, neocortex.config.hidden_dim)
    
    # Store in memory
    stm_state = neocortex.stm.init_state()
    for i in range(5):
        stm_out, stm_state = neocortex.stm(sequence[:, i:i+1], stm_state)
        print(f"Memory Step {i}: STM output shape: {stm_out.shape}")
    
    # Test recall
    ltm_out = neocortex.ltm(sequence)
    print(f"LTM recall shape: {ltm_out.shape}")

def test_attention_mechanism(neocortex: DigitalNeocortex, monitor: NeocortexMonitor):
    """Test attention patterns"""
    print("\nTesting Attention Mechanism...")
    
    # Generate query and key sequences
    query = torch.randn(1, 10, neocortex.config.hidden_dim)
    key = torch.randn(1, 10, neocortex.config.hidden_dim)
    
    # Test attention
    for layer in neocortex.hierarchical_layers:
        attention_output = layer.attention(query, key)
        print(f"Attention output shape: {attention_output.shape}")

def main():
    # Initialize with consistent dimensions
    config = NeocortexConfig(
        input_dim=64,      # Match with hidden_dim
        hidden_dim=64,     # Base dimension for all processing
        num_layers=4,
        num_columns=2,
        memory_capacity=64,  # Match with hidden_dim
        num_heads=4,
        dropout=0.1,
        learning_rate=0.001,
        action_dim=10
    )
    
    # Initialize neocortex and monitor
    neocortex = DigitalNeocortex(config)
    monitor = NeocortexMonitor(neocortex)
    
    # Generate complex inputs
    inputs = generate_complex_inputs(config)
    
    # Test processing with monitoring
    print("Testing Complex Processing...")
    for modality, input_data in inputs.items():
        print(f"\nProcessing {modality}...")
        try:
            output = neocortex.process_input(input_data, modality)
            print(f"Success! Output shape: {output.shape}")
            
            # Get attention weights with matching sequence length
            attention_output = neocortex.hierarchical_layers[0].attention(
                output,
                output
            )
            
            # Record internal states
            monitor.record_state(
                output,
                attention_output,
                neocortex.stm.memory_buffer,
                torch.mean((output[:, :input_data.size(1)] - input_data) ** 2).item()
            )
        except Exception as e:
            print(f"Error processing {modality}: {str(e)}")
    
    # Test specific components
    test_memory_system(neocortex, monitor)
    test_attention_mechanism(neocortex, monitor)
    
    # Visualize results
    monitor.visualize_processing()

if __name__ == "__main__":
    main() 