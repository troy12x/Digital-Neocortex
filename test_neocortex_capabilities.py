import torch
import torch.nn as nn
import numpy as np
from test_reasoning import NeocortexReasoning
import matplotlib.pyplot as plt

class TemporalPatternTest:
    def __init__(self, sequence_length=5, pattern_size=64):
        self.sequence_length = sequence_length
        self.pattern_size = pattern_size
        self.model = NeocortexReasoning(base_features=64)
    
    def generate_complex_temporal_sequence(self):
        """Generate complex overlapping temporal patterns"""
        sequence = []
        center = self.pattern_size // 2
        
        for t in range(self.sequence_length):
            pattern = torch.zeros((1, 1, self.pattern_size, self.pattern_size))
            
            # Pattern 1: Pulsing spiral
            for theta in np.linspace(0, 8*np.pi, 200):
                r = theta * self.pattern_size/32
                x = int(center + r * np.cos(theta + t*np.pi/4))
                y = int(center + r * np.sin(theta + t*np.pi/4))
                if 0 <= x < self.pattern_size and 0 <= y < self.pattern_size:
                    pattern[0, 0, x, y] = np.sin(theta - t*np.pi/2)
            
            # Pattern 2: Expanding/contracting rings
            for i in range(self.pattern_size):
                for j in range(self.pattern_size):
                    dist = np.sqrt((i - center)**2 + (j - center)**2)
                    ring_pattern = np.sin(dist/8 - t*np.pi/3)
                    if pattern[0, 0, i, j] == 0:  # Only add where spiral isn't
                        pattern[0, 0, i, j] = max(0, ring_pattern)
            
            # Pattern 3: Moving wave interference
            for i in range(self.pattern_size):
                for j in range(self.pattern_size):
                    wave1 = np.sin(i/8 + t*np.pi/4)
                    wave2 = np.sin(j/8 - t*np.pi/4)
                    interference = (wave1 + wave2) / 2
                    pattern[0, 0, i, j] += interference * 0.3  # Add interference pattern
            
            # Normalize pattern
            pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
            sequence.append(pattern)
        
        return torch.cat(sequence, dim=0)

    def test_temporal_prediction(self):
        """Test model's ability to predict complex temporal patterns"""
        # Generate temporal sequence
        sequence = self.generate_complex_temporal_sequence()
        
        # Split into input and target
        input_seq = sequence[:-1]  # All but last
        target = sequence[-1:]     # Last frame
        
        # Train model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        print("Training on complex temporal sequence...")
        losses = []
        for epoch in range(200):  # More epochs for complex patterns
            optimizer.zero_grad()
            
            # Forward pass with multiple inputs for temporal context
            prediction = self.model(input_seq[-2], input_seq[-1])  # Use last two frames
            loss = criterion(prediction, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        return prediction, target, loss.item(), losses

    def visualize_results(self, prediction, target, loss, losses=None):
        """Visualize the temporal prediction results with enhanced plotting"""
        plt.figure(figsize=(15, 5))
        
        # Plot target pattern
        plt.subplot(131)
        plt.imshow(target.squeeze().detach().numpy(), cmap='plasma')
        plt.title('Target Pattern')
        plt.axis('off')
        
        # Plot predicted pattern
        plt.subplot(132)
        plt.imshow(prediction.squeeze().detach().numpy(), cmap='plasma')
        plt.title(f'Predicted Pattern\nLoss: {loss:.4f}')
        plt.axis('off')
        
        # Plot loss curve if available
        if losses is not None:
            plt.subplot(133)
            plt.plot(losses)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.yscale('log')
        
        plt.suptitle('Complex Temporal Pattern Prediction')
        plt.tight_layout()
        plt.show()

def test_hierarchical_learning():
    """Test hierarchical pattern learning capabilities"""
    model = NeocortexReasoning(base_features=64)
    
    # Create patterns with different levels of complexity
    patterns = []
    size = 64
    
    # Level 1: Simple geometric pattern
    pattern1 = torch.zeros((1, 1, size, size))
    center = size // 2
    radius = size // 4
    for i in range(size):
        for j in range(size):
            if (i - center)**2 + (j - center)**2 <= radius**2:
                pattern1[0, 0, i, j] = 1.0
    
    # Level 2: Add frequency components
    pattern2 = pattern1.clone()
    for i in range(size):
        for j in range(size):
            if pattern2[0, 0, i, j] > 0:
                pattern2[0, 0, i, j] *= np.sin(i/5) * np.cos(j/5)
    
    # Level 3: Add spatial complexity
    pattern3 = pattern2.clone()
    for i in range(size):
        for j in range(size):
            if pattern3[0, 0, i, j] > 0:
                pattern3[0, 0, i, j] *= (1 + 0.5*np.sin(np.sqrt((i-center)**2 + (j-center)**2)/10))
    
    # Test model's response to increasing complexity
    patterns = [pattern1, pattern2, pattern3]
    
    # Visualize hierarchical learning
    plt.figure(figsize=(15, 5))
    for i, pattern in enumerate(patterns):
        plt.subplot(1, 3, i+1)
        plt.imshow(pattern.squeeze(), cmap='viridis')
        plt.title(f'Complexity Level {i+1}')
        plt.axis('off')
    
    plt.suptitle('Hierarchical Pattern Complexity')
    plt.show()

if __name__ == "__main__":
    # Test complex temporal pattern prediction
    temporal_test = TemporalPatternTest(sequence_length=6)  # Longer sequence
    prediction, target, loss, losses = temporal_test.test_temporal_prediction()
    temporal_test.visualize_results(prediction, target, loss, losses)
    
    # Test hierarchical learning
    test_hierarchical_learning()
