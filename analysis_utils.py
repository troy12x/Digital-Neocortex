import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os

class NeocortexAnalyzer:
    def __init__(self, neocortex, test_patterns):
        self.neocortex = neocortex
        self.test_patterns = test_patterns
        self.device = next(neocortex.parameters()).device
        # Get config from neocortex if available, otherwise use default
        self.config = getattr(neocortex, 'config', None)
        self.hidden_dim = getattr(self.config, 'hidden_dim', 64)  # Default to 64 if not specified
        
        # Create directories if they don't exist
        Path("models").mkdir(exist_ok=True)
        Path("visualizations").mkdir(exist_ok=True)
        
    def save_models(self):
        """Save trained models"""
        print("\nSaving models...")
        torch.save(self.neocortex.state_dict(), 'models/neocortex.pth')
        torch.save(self.test_patterns.baseline_rnn.state_dict(), 'models/rnn.pth')
        torch.save(self.test_patterns.baseline_transformer.state_dict(), 'models/transformer.pth')
        print("Models saved successfully!")
        
    def visualize_patterns(self):
        """Visualize original and reconstructed patterns"""
        patterns = self.test_patterns.generate_test_patterns()
        
        for name, pattern in patterns.items():
            print(f"\n{name}:")
            
            # Process through neocortex
            with torch.no_grad():
                reconstruction = self.neocortex(pattern, modality='pattern')
            
            # Ensure proper dimensions
            if pattern.shape != reconstruction.shape:
                reconstruction = reconstruction.view(1, 1, 32, 32)
            
            # Convert to numpy for plotting
            pattern_np = pattern.squeeze().cpu().numpy()
            reconstruction_np = reconstruction.squeeze().cpu().numpy()
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            # Plot original pattern
            im1 = ax1.imshow(pattern_np, cmap='viridis')
            ax1.set_title('Original Pattern')
            plt.colorbar(im1, ax=ax1)
            
            # Plot reconstruction
            im2 = ax2.imshow(reconstruction_np, cmap='viridis')
            ax2.set_title('Reconstructed Pattern')
            plt.colorbar(im2, ax=ax2)
            
            # Save plot
            plt.savefig(f'visualizations/{name}_comparison.png')
            plt.close()
            
            # Calculate and print metrics
            similarity = self.calculate_similarity(pattern, reconstruction)
            error = torch.mean((reconstruction - pattern) ** 2).item()
            
            print(f"Similarity: {similarity:.4f}")
            print(f"Error: {error:.4f}")
    
    def analyze_attention(self, pattern_type, pattern, reconstruction):
        """Analyze attention patterns for different pattern types"""
        device = pattern.device
        
        # Get model's attention weights
        attention_weights = {}
        
        def attention_hook(name):
            def hook(module, input, output):
                # For nn.MultiheadAttention, output is (attn_output, attn_weights)
                if isinstance(output, tuple) and len(output) > 1:
                    attention_weights[name] = output[1].detach()
            return hook
        
        # Register hooks for all attention layers
        hooks = []
        
        # Cross-scale attention
        hooks.append(self.neocortex.pattern_processor.cross_scale_attention.register_forward_hook(
            attention_hook("cross_scale_attention")
        ))
        
        # Run forward pass to get attention weights
        with torch.no_grad():
            _ = self.neocortex(pattern, modality='pattern')
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Visualize attention patterns
        for name, weights in attention_weights.items():
            # Reshape attention weights if needed
            if weights.dim() == 4:  # [batch, heads, seq_len, seq_len]
                weights = weights.mean(1)  # Average over heads
            
            # Create visualization directory if it doesn't exist
            Path("visualizations/attention").mkdir(parents=True, exist_ok=True)
            
            # Save attention visualization
            save_path = f"visualizations/attention/{pattern_type}_{name}.png"
            self.visualize_attention(weights, name, save_path)
        
        return attention_weights

    def visualize_attention(self, attention_weights, layer_name, save_path):
        """Visualize attention patterns with better zoom and clarity"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention_weights[0].cpu().numpy(),
            cmap='viridis',
            xticklabels=False,
            yticklabels=False
        )
        plt.title(f'Attention Pattern: {layer_name}')
        plt.savefig(save_path)
        plt.close()

    def analyze_pattern(self, pattern_type):
        """Analyze a specific pattern type"""
        pattern = self.generate_pattern(pattern_type)
        
        # Process through model
        with torch.no_grad():
            reconstruction = self.neocortex(pattern, modality='pattern')
        
        # Calculate similarity and error
        similarity = self.calculate_similarity(pattern, reconstruction)
        error = F.mse_loss(reconstruction, pattern)
        
        print(f"\n{pattern_type}:")
        print(f"Similarity: {similarity:.4f}")
        print(f"Error: {error:.4f}")
        
        # Analyze attention patterns
        attention_analysis = self.analyze_attention(pattern_type, pattern, reconstruction)
        
        return {
            'pattern': pattern,
            'reconstruction': reconstruction,
            'similarity': similarity,
            'error': error.item(),
            'attention_analysis': attention_analysis
        }
    
    def test_new_patterns(self):
        """Test the model on new patterns"""
        print("\nTesting New Patterns:")
        print("=" * 50)
        
        # Generate test patterns
        test_patterns = self.test_patterns.generate_test_patterns()
        
        for pattern_type, pattern in test_patterns.items():
            print(f"\n{pattern_type}:")
            
            # Process through neocortex
            with torch.no_grad():
                reconstruction = self.neocortex(pattern, modality='pattern')
            
            # Ensure proper dimensions
            if pattern.shape != reconstruction.shape:
                reconstruction = reconstruction.view(1, 1, 32, 32)
            
            # Calculate metrics
            similarity = self.calculate_similarity(pattern, reconstruction)
            error = F.mse_loss(reconstruction, pattern)
            
            # Create visualization
            plt.figure(figsize=(12, 4))
            
            # Plot original
            plt.subplot(1, 2, 1)
            plt.imshow(pattern.squeeze().cpu().numpy(), cmap='viridis')
            plt.title('Original Pattern')
            plt.colorbar()
            
            # Plot reconstruction
            plt.subplot(1, 2, 2)
            plt.imshow(reconstruction.squeeze().cpu().numpy(), cmap='viridis')
            plt.title('Reconstruction')
            plt.colorbar()
            
            # Save plot
            os.makedirs('visualizations/new_patterns', exist_ok=True)
            plt.savefig(f'visualizations/new_patterns/{pattern_type}_comparison.png')
            plt.close()
            
            # Print metrics
            print(f"Similarity: {similarity:.4f}")
            print(f"Error: {error.item():.4f}")
    
    def generate_pattern(self, pattern_type):
        """Generate different types of patterns for analysis."""
        if pattern_type == 'simple':
            # Generate a simple 32x32 pattern
            t = torch.linspace(0, 2*np.pi, 32)
            x, y = torch.meshgrid(t, t)
            pattern = torch.sin(x) * torch.cos(y)
            pattern = pattern.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims [1, 1, 32, 32]
            pattern = pattern.repeat(1, self.hidden_dim, 1, 1)  # Expand channels correctly [1, 64, 32, 32]
            return pattern.to(self.device)
        elif pattern_type == 'burst':
            # Generate a burst pattern
            pattern = torch.zeros(1, self.hidden_dim, 32, 32)  # Correct dimension order
            pattern[:, :, 10:20, 10:20] = 1.0
            return pattern.to(self.device)
        elif pattern_type == 'am_signal':
            # Generate an amplitude modulated signal
            t = torch.linspace(0, 4*np.pi, 32)
            carrier = torch.sin(8*t)
            modulator = 0.5 * (1 + torch.sin(t))
            signal = carrier * modulator
            pattern = signal.view(1, 1, 32, 1).repeat(1, self.hidden_dim, 1, 32)  # Correct dimension order
            return pattern.to(self.device)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

    def normalize_pattern(self, pattern):
        """Normalize pattern with robust scaling"""
        # Calculate robust quantiles for scaling
        q_low, q_high = torch.quantile(pattern, torch.tensor([0.05, 0.95], device=pattern.device))
        
        # Clip and scale to [0, 1] range
        pattern_norm = torch.clamp(pattern, q_low, q_high)
        pattern_norm = (pattern_norm - q_low) / (q_high - q_low + 1e-8)
        
        return pattern_norm

    def calculate_similarity(self, pattern, reconstruction):
        """Calculate similarity using SSIM, gradient, and frequency domain metrics"""
        device = pattern.device
        
        if pattern.device != reconstruction.device:
            reconstruction = reconstruction.to(device)
        
        # Ensure 4D tensors [B, C, H, W]
        if len(pattern.shape) == 2:
            pattern = pattern.view(1, 1, *pattern.shape)
        elif len(pattern.shape) == 3:
            pattern = pattern.unsqueeze(1)
            
        if len(reconstruction.shape) == 2:
            reconstruction = reconstruction.view(1, 1, *reconstruction.shape)
        elif len(reconstruction.shape) == 3:
            reconstruction = reconstruction.unsqueeze(1)
        
        if pattern.shape != reconstruction.shape:
            reconstruction = F.interpolate(reconstruction, size=pattern.shape[-2:], mode='bilinear', align_corners=False)
            
        # Apply robust normalization to both patterns
        pattern = self.normalize_pattern(pattern)
        reconstruction = self.normalize_pattern(reconstruction)
        
        # 1. Calculate SSIM
        def calculate_ssim(x, y):
            C1, C2 = (0.01 * 255)**2, (0.03 * 255)**2
            
            # Calculate means with 11x11 kernel
            mu_x = F.avg_pool2d(x, kernel_size=11, stride=1, padding=5)
            mu_y = F.avg_pool2d(y, kernel_size=11, stride=1, padding=5)
            
            # Calculate variances and covariance
            sigma_x = F.avg_pool2d(x**2, kernel_size=11, stride=1, padding=5) - mu_x**2
            sigma_y = F.avg_pool2d(y**2, kernel_size=11, stride=1, padding=5) - mu_y**2
            sigma_xy = F.avg_pool2d(x*y, kernel_size=11, stride=1, padding=5) - mu_x*mu_y
            
            ssim = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2))
            return ssim.mean()
        
        # 2. Calculate gradient similarity
        def calculate_gradient_sim(x, y):
            # Sobel filters
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device).float().view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device).float().view(1, 1, 3, 3)
            
            # Handle multi-channel inputs
            if x.size(1) > 1:
                x = x.mean(dim=1, keepdim=True)
            if y.size(1) > 1:
                y = y.mean(dim=1, keepdim=True)
            
            # Calculate gradients
            grad_x_x = F.conv2d(x, sobel_x, padding=1)
            grad_x_y = F.conv2d(x, sobel_y, padding=1)
            grad_y_x = F.conv2d(y, sobel_x, padding=1)
            grad_y_y = F.conv2d(y, sobel_y, padding=1)
            
            # Calculate gradient magnitudes
            grad_mag_x = torch.sqrt(grad_x_x**2 + grad_x_y**2 + 1e-6)
            grad_mag_y = torch.sqrt(grad_y_x**2 + grad_y_y**2 + 1e-6)
            
            return F.cosine_similarity(grad_mag_x.flatten(), grad_mag_y.flatten(), dim=0)
        
        # 3. Calculate frequency domain similarity
        def calculate_freq_sim(x, y):
            freq_x = torch.fft.fft2(x)
            freq_y = torch.fft.fft2(y)
            
            mag_x = torch.abs(freq_x)
            mag_y = torch.abs(freq_y)
            
            return F.cosine_similarity(mag_x.flatten(), mag_y.flatten(), dim=0)
        
        # Calculate individual components
        ssim = calculate_ssim(pattern, reconstruction)
        grad_sim = calculate_gradient_sim(pattern, reconstruction)
        freq_sim = calculate_freq_sim(pattern, reconstruction)
        
        # Combine metrics with weights
        similarity = (
            0.4 * ssim +          # Structure preservation
            0.3 * grad_sim +      # Edge preservation
            0.3 * freq_sim        # Frequency content
        )
        
        return similarity.item()

def main():
    from test_neocortex import TestPatterns
    
    # Load test patterns and models
    test_patterns = TestPatterns()
    
    # Create analyzer
    analyzer = NeocortexAnalyzer(test_patterns.neocortex, test_patterns)
    
    # Run analysis
    analyzer.save_models()
    analyzer.visualize_patterns()
    analyzer.analyze_attention('simple', analyzer.generate_pattern('simple'), analyzer.neocortex(analyzer.generate_pattern('simple'), modality='pattern'))
    analyzer.test_new_patterns()
    
    print("\nAnalysis complete! Check the 'visualizations' directory for results.")

if __name__ == "__main__":
    main()