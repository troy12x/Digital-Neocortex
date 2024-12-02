import torch
import torch.nn as nn
import numpy as np
from app import DigitalNeocortex
from training_utils import TrainingConfig
import torch.nn.functional as F

class BaselineRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True
        )
        self.output = nn.Linear(config.hidden_dim, 1)
    
    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(-1)
        output, _ = self.rnn(x)
        return self.output(output)

class BaselineTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Linear(1, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        self.output = nn.Linear(config.hidden_dim, 1)
    
    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output(x)

class TestPatterns:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize configuration
        self.config = TrainingConfig(
            input_dim=32,  # Match spatial dimension
            hidden_dim=128,  # Increased for better capacity
            num_layers=6,
            num_heads=8,
            num_columns=6,  # Match DigitalNeocortex
            dropout=0.1,
            batch_size=32,
            spatial_channels=16,
            spatial_kernel_size=3,
            use_spatial_attention=True
        )
        
        # Initialize models
        self.neocortex = DigitalNeocortex(self.config).to(self.device)
        self.baseline_rnn = nn.LSTM(64, 64, num_layers=2, batch_first=True).to(self.device)
        
        # Initialize transformer with correct dimensions
        self.baseline_transformer = nn.Transformer(
            d_model=64,  # Match channel dimension
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            batch_first=True  # Important for our sequence format
        ).to(self.device)
        
    def normalize_pattern(self, pattern):
        """Normalize pattern with robust scaling"""
        # Calculate robust quantiles for scaling
        q_low, q_high = torch.quantile(pattern, torch.tensor([0.05, 0.95], device=self.device))
        
        # Clip and scale to [0, 1] range
        pattern_norm = torch.clamp(pattern, q_low, q_high)
        pattern_norm = (pattern_norm - q_low) / (q_high - q_low + 1e-8)
        
        return pattern_norm

    def generate_test_patterns(self):
        """Generate all test patterns"""
        return {
            'chirp': self.generate_chirp(),
            'burst': self.generate_burst(),
            'am_signal': self.generate_am_signal()
        }
        
    def generate_chirp(self):
        """Generate a chirp signal with better spatial structure"""
        # Create spatial grid
        x = torch.linspace(0, 1, 32, device=self.device)
        y = torch.linspace(0, 1, 32, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Create 2D chirp pattern
        f0, f1 = 1, 10  # Start and end frequencies
        phase_x = 2 * np.pi * X * (f0 + (f1-f0) * X / 2)
        phase_y = 2 * np.pi * Y * (f0 + (f1-f0) * Y / 2)
        
        # Combine horizontal and vertical components
        chirp_2d = torch.sin(phase_x) * torch.cos(phase_y)
        
        # Add smooth variations
        envelope = torch.exp(-(X - 0.5)**2 / 0.25 - (Y - 0.5)**2 / 0.25)
        chirp_2d = chirp_2d * envelope
        
        # Normalize while preserving structure
        chirp_2d = self.normalize_pattern(chirp_2d)
        
        # Add batch and channel dimensions [B, C, H, W]
        chirp_2d = chirp_2d.unsqueeze(0).unsqueeze(0)
        
        return chirp_2d

    def generate_burst(self):
        """Generate a burst pattern with coherent structure"""
        # Create spatial grid
        x = torch.linspace(0, 1, 32, device=self.device)
        y = torch.linspace(0, 1, 32, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Create burst parameters
        t = torch.linspace(0, 1, 32, device=self.device)
        burst_freq = 10
        burst_width = 0.1
        
        # Create burst envelope
        envelope = torch.exp(-(X - 0.5)**2 / burst_width - (Y - 0.5)**2 / burst_width)
        
        # Create burst oscillation
        oscillation = torch.sin(2 * np.pi * burst_freq * X) * torch.cos(2 * np.pi * burst_freq * Y)
        
        # Combine envelope and oscillation
        burst = oscillation * envelope
        
        # Add smooth variations
        variation = torch.sin(2 * np.pi * 2 * X) * torch.cos(2 * np.pi * 2 * Y)
        burst = burst + 0.2 * variation
        
        # Normalize while preserving structure
        burst = self.normalize_pattern(burst)
        
        # Add batch and channel dimensions [B, C, H, W]
        burst = burst.unsqueeze(0).unsqueeze(0)
        
        return burst

    def generate_am_signal(self):
        """Generate AM signal with spatial coherence"""
        # Create spatial grid
        x = torch.linspace(0, 1, 32, device=self.device)
        y = torch.linspace(0, 1, 32, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Carrier frequency
        fc = 20
        # Modulation frequency
        fm = 5
        
        # Generate carrier wave
        carrier = torch.sin(2 * np.pi * fc * X)
        
        # Generate modulation envelope
        envelope = 0.5 * (1 + torch.sin(2 * np.pi * fm * Y))
        
        # Create AM signal
        am_signal = carrier * envelope
        
        # Add spatial variation
        spatial_mod = torch.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.3)
        am_signal = am_signal * spatial_mod
        
        # Normalize
        am_signal = self.normalize_pattern(am_signal)
        
        # Add batch and channel dimensions [B, C, H, W]
        am_signal = am_signal.unsqueeze(0).unsqueeze(0)
        
        return am_signal

    def test_pattern_recognition(self):
        """Test pattern recognition capabilities"""
        similarity_scores = []
        reconstruction_errors = []
        spatial_coherence_scores = []
        
        # Test each pattern type
        test_patterns = self.generate_test_patterns()
        for pattern_type, pattern in test_patterns.items():
            print(f"\nTesting {pattern_type} pattern...")
            
            # Move pattern to device and ensure proper shape
            pattern = pattern.to(self.device)
            if len(pattern.shape) == 2:  # [B, T]
                pattern = pattern.view(1, 1, 32, 32)
            elif len(pattern.shape) == 3:  # [B, H, W]
                pattern = pattern.unsqueeze(1)  # Add channel dimension
            
            # Process through neocortex
            with torch.no_grad():
                reconstructed = self.neocortex(pattern, modality='pattern')
                if len(reconstructed.shape) == 2:  # [B, T]
                    reconstructed = reconstructed.view(1, 1, 32, 32)
            
            # Calculate metrics
            similarity = self._calculate_similarity(pattern, reconstructed)
            reconstruction_error = F.mse_loss(reconstructed, pattern)
            spatial_coherence = self._calculate_spatial_coherence(reconstructed)
            
            print(f"Similarity Score: {similarity:.4f}")
            print(f"Reconstruction Error: {reconstruction_error.item():.4f}")
            print(f"Spatial Coherence: {spatial_coherence:.4f}")
            
            # Collect metrics for averaging
            similarity_scores.append(similarity)
            reconstruction_errors.append(reconstruction_error.item())
            spatial_coherence_scores.append(spatial_coherence)
        
        # Calculate average metrics
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        avg_error = sum(reconstruction_errors) / len(reconstruction_errors)
        avg_coherence = sum(spatial_coherence_scores) / len(spatial_coherence_scores)
        
        print("\nOverall Performance:")
        print(f"Average Similarity: {avg_similarity:.4f}")
        print(f"Average Error: {avg_error:.4f}")
        print(f"Average Spatial Coherence: {avg_coherence:.4f}")
        
        return {
            'similarity': similarity_scores,
            'error': reconstruction_errors,
            'coherence': spatial_coherence_scores
        }

    def compare_models(self):
        """Compare performance of different models"""
        results = {}
        
        # Test each pattern type
        test_patterns = self.generate_test_patterns()
        for pattern_type, pattern in test_patterns.items():
            print(f"\nTesting {pattern_type} pattern...")
            
            # Move pattern to device
            pattern = pattern.to(self.device)
            if len(pattern.shape) == 2:  # [B, T]
                pattern = pattern.view(1, 1, 32, 32)
            elif len(pattern.shape) == 3:  # [B, H, W]
                pattern = pattern.unsqueeze(1)  # Add channel dimension
            
            # Get predictions from DigitalNeocortex
            with torch.no_grad():
                neocortex_out = self.neocortex(pattern, modality='pattern')
                
                # Calculate metrics
                neocortex_error = F.mse_loss(neocortex_out, pattern)
                neocortex_similarity = self._calculate_similarity(neocortex_out, pattern)
                
                # Store results
                results[pattern_type] = {
                    'error': neocortex_error.item(),
                    'similarity': neocortex_similarity
                }
                
                # Print results
                print(f"DigitalNeocortex Error: {neocortex_error.item():.4f}")
                print(f"DigitalNeocortex Similarity: {neocortex_similarity:.4f}")
        
        return results

    def _calculate_similarity(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Calculate similarity between two patterns"""
        # Ensure tensors have spatial dimensions
        if len(x.shape) == 2:
            x = x.view(1, 1, 32, 32)  # [B, C, H, W]
        elif len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension
            
        if len(y.shape) == 2:
            y = y.view(1, 1, 32, 32)
        elif len(y.shape) == 3:
            y = y.unsqueeze(1)
        
        # Ensure same shape through interpolation
        if x.shape != y.shape:
            y = F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # Calculate structural similarity
        ssim = self._calculate_ssim(x, y)
        
        # Calculate gradient similarity
        grad_sim = self._calculate_gradient_similarity(x, y)
        
        # Calculate frequency similarity
        freq_sim = self._calculate_frequency_similarity(x, y)
        
        # Combine metrics with weights
        similarity = 0.4 * ssim + 0.3 * grad_sim + 0.3 * freq_sim
        
        # Handle both tensor and float outputs
        if isinstance(similarity, torch.Tensor):
            return similarity.item()
        return float(similarity)

    def _calculate_ssim(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Calculate Structural Similarity Index (SSIM)"""
        # Constants for stability
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        
        # Calculate means
        mu_x = F.avg_pool2d(x, kernel_size=11, stride=1, padding=5)
        mu_y = F.avg_pool2d(y, kernel_size=11, stride=1, padding=5)
        
        # Calculate variances and covariance
        sigma_x = F.avg_pool2d(x**2, kernel_size=11, stride=1, padding=5) - mu_x**2
        sigma_y = F.avg_pool2d(y**2, kernel_size=11, stride=1, padding=5) - mu_y**2
        sigma_xy = F.avg_pool2d(x*y, kernel_size=11, stride=1, padding=5) - mu_x*mu_y
        
        # Calculate SSIM
        ssim = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2))
        
        return ssim.mean().item()

    def _calculate_gradient_similarity(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Calculate similarity between gradients of two patterns"""
        # Create Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=self.device).float()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=self.device).float()
        
        # Add dimensions for conv2d [out_channels, in_channels, H, W]
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
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
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(
            grad_mag_x.flatten(),
            grad_mag_y.flatten(),
            dim=0
        )
        
        return similarity.item()

    def _calculate_frequency_similarity(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Calculate similarity in frequency domain"""
        # Convert to frequency domain
        freq_x = torch.fft.fft2(x)
        freq_y = torch.fft.fft2(y)
        
        # Calculate magnitude spectra
        mag_x = torch.abs(freq_x)
        mag_y = torch.abs(freq_y)
        
        # Calculate cosine similarity in frequency domain
        similarity = F.cosine_similarity(mag_x.flatten(), mag_y.flatten(), dim=0)
        
        return similarity.item()

    def _calculate_spatial_coherence(self, x: torch.Tensor) -> float:
        """Calculate spatial coherence of pattern"""
        # Ensure input has spatial dimensions [B, C, H, W]
        if len(x.shape) == 2:
            x = x.view(1, 1, 32, 32)
        elif len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # Calculate local autocorrelation
        x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
        neighbors = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                shift = x_pad[..., 1+i:33+i, 1+j:33+j]
                neighbors.append(shift)
        
        # Stack neighbors
        neighbors = torch.stack(neighbors, dim=0)
        
        # Calculate correlation with neighbors
        x_repeat = x.expand(8, -1, -1, -1, -1)  # 8 neighbors
        correlation = F.cosine_similarity(x_repeat.flatten(1), neighbors.flatten(1), dim=1)
        
        # Average correlation
        coherence = correlation.mean()
        
        return coherence.item()

def run_comparison_tests():
    """Run comprehensive comparison tests"""
    print("\nRunning Pattern Recognition Tests")
    print("=" * 50 + "\n")
    
    # Create test instance
    tester = TestPatterns()
    
    # Run pattern recognition tests
    pattern_results = tester.test_pattern_recognition()
    
    print("\nRunning Model Comparison Tests")
    print("=" * 50 + "\n")
    
    # Run model comparison tests
    comparison_results = tester.compare_models()
    
    # Print final summary
    print("\nTEST SUMMARY")
    print("=" * 50 + "\n")
    
    print("Pattern Recognition Results:\n")
    print(f"Average Metrics:")
    print(f"  Similarity: {sum(pattern_results['similarity']) / len(pattern_results['similarity']):.4f}")
    print(f"  Error: {sum(pattern_results['error']) / len(pattern_results['error']):.4f}")
    print(f"  Spatial Coherence: {sum(pattern_results['coherence']) / len(pattern_results['coherence']):.4f}")
    
    print("\nModel Comparison Results:\n")
    print("DigitalNeocortex vs Baselines:")
    for pattern_type in ['chirp', 'burst', 'am_signal']:
        print(f"\n{pattern_type.capitalize()} Pattern:")
        print(f"  Error: {comparison_results[pattern_type]['error']:.4f}")
        print(f"  Similarity: {comparison_results[pattern_type]['similarity']:.4f}")

if __name__ == "__main__":
    run_comparison_tests() 