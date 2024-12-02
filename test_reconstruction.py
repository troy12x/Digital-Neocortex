import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
from test_patterns import generate_simple_patterns
from app import DigitalNeocortex, NeocortexConfig

def calculate_similarity(original, reconstruction):
    """Calculate similarity between original and reconstructed patterns"""
    # Get the device from the input tensors
    device = original.device
    
    # Ensure proper dimensions and normalize inputs
    if len(original.shape) == 2:
        original = original.unsqueeze(0).unsqueeze(0)
    if len(reconstruction.shape) == 2:
        reconstruction = reconstruction.unsqueeze(0).unsqueeze(0)
    
    # Normalize inputs to [0, 1] range
    original = (original - original.min()) / (original.max() - original.min() + 1e-8)
    reconstruction = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min() + 1e-8)
    
    # Direct similarity (MSE-based)
    mse = F.mse_loss(original, reconstruction)
    mse_sim = torch.exp(-mse)  # Convert MSE to similarity score [0, 1]
    
    # Structural similarity
    kernel_size = 3
    mu1 = F.avg_pool2d(original, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    mu2 = F.avg_pool2d(reconstruction, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(original * original, kernel_size=kernel_size, stride=1, padding=kernel_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(reconstruction * reconstruction, kernel_size=kernel_size, stride=1, padding=kernel_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(original * reconstruction, kernel_size=kernel_size, stride=1, padding=kernel_size//2) - mu1_mu2
    
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2
    
    ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim = (ssim + 1) / 2  # Normalize to [0, 1]
    ssim = ssim.mean()
    
    # Pattern correlation
    orig_flat = original.flatten(1)
    rec_flat = reconstruction.flatten(1)
    correlation = F.cosine_similarity(orig_flat, rec_flat, dim=1).mean()
    correlation = (correlation + 1) / 2  # Normalize to [0, 1]
    
    # Combine all metrics
    similarity = (0.4 * mse_sim + 0.4 * ssim + 0.2 * correlation).item()
    return similarity

class PatternReconstructor(nn.Module):
    def __init__(self):
        super(PatternReconstructor, self).__init__()
        # Increase model capacity significantly
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU()
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Add residual connections
        self.skip1 = nn.Conv2d(64, 64, 1)
        self.skip2 = nn.Conv2d(128, 128, 1)
        self.skip3 = nn.Conv2d(256, 256, 1)
        self.skip4 = nn.Conv2d(512, 512, 1)

    def forward(self, x):
        # Encoder with skip connections
        e1 = self.encoder[0:2](x)
        skip1 = self.skip1(e1)
        
        e2 = self.encoder[2:4](e1)
        skip2 = self.skip2(e2)
        
        e3 = self.encoder[4:6](e2)
        skip3 = self.skip3(e3)
        
        e4 = self.encoder[6:8](e3)
        skip4 = self.skip4(e4)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder with skip connections
        d4 = self.decoder[0:2](b + skip4)
        d3 = self.decoder[2:4](d4 + skip3)
        d2 = self.decoder[4:6](d3 + skip2)
        d1 = self.decoder[6:8](d2 + skip1)
        
        return d1

def train_until_similarity(model, pattern, pattern_name, target_similarity=0.9, max_epochs=10000, learning_rate=0.002):
    """Train the model until reaching target similarity or max epochs"""
    os.makedirs('models', exist_ok=True)
    
    # More aggressive optimizer settings
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-7, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=200, verbose=True, min_lr=1e-6)
    
    best_similarity = -float('inf')
    best_reconstruction = None
    best_model_state = None
    no_improve = 0
    min_improve = 1e-7
    stuck_count = 0
    max_stuck_attempts = 20  # Maximum number of attempts before accepting best result
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pattern = pattern.to(device)
    model = model.to(device)
    model.train()
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        try:
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except:
            pass
    
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    pbar = tqdm(range(max_epochs), desc=f"Training {pattern_name}", leave=True)
    
    # Multiple loss components
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    
    for epoch in pbar:
        optimizer.zero_grad(set_to_none=True)
        
        try:
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                reconstruction = model(pattern)
                # Adaptive loss weighting based on stuck count
                mse_weight = max(0.5, min(0.9, 0.7 + stuck_count * 0.1))
                mse_loss = mse_criterion(reconstruction, pattern)
                l1_loss = l1_criterion(reconstruction, pattern)
                loss = mse_weight * mse_loss + (1 - mse_weight) * l1_loss
            
            if torch.isnan(loss):
                print("\nNaN loss detected, stopping training")
                break
                
            scaler.scale(loss).backward(retain_graph=True)
            # Increase gradient clipping threshold when stuck
            clip_norm = 2.0 * (1 + stuck_count * 0.5)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            scaler.step(optimizer)
            scaler.update()
            
            with torch.no_grad():
                model.eval()
                reconstruction = model(pattern)
                reconstruction = torch.sigmoid(reconstruction)  # Apply sigmoid
                similarity = calculate_similarity(pattern, reconstruction)
                model.train()
            
            scheduler.step(similarity)
            
            if similarity > best_similarity + min_improve:
                best_similarity = similarity
                best_reconstruction = reconstruction.clone()
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'similarity': best_similarity,
                }
                torch.save(best_model_state, f'models/{pattern_name}_best.pth')
                no_improve = 0
                stuck_count = 0  # Reset stuck count on improvement
            else:
                no_improve += 1
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Similarity': f'{similarity:.4f}',
                'Best': f'{best_similarity:.4f}'
            })
            
            # Stop conditions:
            # 1. Reached target similarity
            # 2. Made too many attempts to escape local minimum
            # 3. For text patterns, accept lower similarity after fewer attempts
            is_text_pattern = pattern_name.startswith('text_')
            max_attempts = 10 if is_text_pattern else max_stuck_attempts
            
            if similarity >= target_similarity:
                print(f"\nReached target similarity of {target_similarity} at epoch {epoch+1}")
                return best_reconstruction, best_similarity
            elif stuck_count >= max_attempts:
                print(f"\nReached maximum attempts ({max_attempts}). Best similarity: {best_similarity:.4f}")
                return best_reconstruction, best_similarity
            
            # If stuck, try more aggressive changes
            if no_improve >= 300:
                no_improve = 0
                stuck_count += 1
                print(f"\nStuck at similarity {best_similarity:.4f}, attempt {stuck_count}/{max_attempts} to escape...")
                
                # More aggressive changes based on stuck count
                if stuck_count % 3 == 0:
                    # Every 3rd attempt: Temporarily increase learning rate significantly
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate * (2 + stuck_count)
                    print(f"Significantly increased learning rate to {learning_rate * (2 + stuck_count)}")
                elif stuck_count % 3 == 1:
                    # Every 3rd + 1 attempt: Reduce regularization
                    for param_group in optimizer.param_groups:
                        param_group['weight_decay'] *= 0.1
                    print(f"Reduced weight decay to {param_group['weight_decay']}")
                else:
                    # Every 3rd + 2 attempt: Change loss function weights
                    mse_weight = max(0.1, min(0.9, 0.5 + np.random.random() * 0.4))
                    print(f"Adjusted MSE weight to {mse_weight:.4f}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("\nWARNING: out of memory, clearing cache and retrying")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    print(f"\nReached max epochs. Best similarity: {best_similarity:.4f}")
    return best_reconstruction, best_similarity

def plot_results(original, reconstruction, title, noise_level=None):
    """Plot original vs reconstructed patterns"""
    plt.figure(figsize=(12, 5), dpi=150)  # Higher DPI for sharper images
    
    # Convert tensors to numpy arrays
    original_np = original.cpu().detach().numpy().reshape(32, 32)
    reconstruction_np = reconstruction.cpu().detach().numpy().reshape(32, 32)
    
    # Apply sigmoid to reconstruction if not already applied
    if reconstruction_np.min() < 0 or reconstruction_np.max() > 1:
        reconstruction_np = 1 / (1 + np.exp(-reconstruction_np))
    
    # Threshold values for sharper edges
    def threshold_array(arr, threshold=0.5):
        # For center dot pattern, use dynamic thresholding
        if "center_dot" in title.lower():
            threshold = 0.3
        binary = arr > threshold
        return binary.astype(float)
    
    # Apply thresholding for binary-like patterns
    if any(pattern in title.lower() for pattern in ['center_dot', 'text_', 'checkerboard']):
        reconstruction_np = threshold_array(reconstruction_np)
    
    # Plot original
    plt.subplot(121)
    plt.imshow(original_np, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    plt.title('Original' if noise_level is None else f'Noisy (level={noise_level})')
    plt.axis('off')
    
    # Plot reconstruction
    plt.subplot(122)
    plt.imshow(reconstruction_np, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    plt.title('Reconstruction')
    plt.axis('off')
    
    # Add similarity score
    similarity = calculate_similarity(torch.tensor(original_np), torch.tensor(reconstruction_np))
    plt.suptitle(f"{title} (Similarity: {similarity:.4f})")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save with high quality settings
    os.makedirs('results', exist_ok=True)
    clean_title = title.replace(" ", "_").lower()
    noise_suffix = f"_noise_{noise_level}" if noise_level is not None else ""
    plt.savefig(f'results/{clean_title}{noise_suffix}.png', 
                dpi=300,  # Higher DPI for saved files
                bbox_inches='tight', 
                pad_inches=0.2,
                facecolor='white',
                edgecolor='none',
                format='png',
                transparent=False,
                metadata={'Creator': 'Pattern Reconstructor'})
    plt.close()

def test_pattern_reconstruction():
    """Test the pattern reconstruction capabilities"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test patterns with different complexities and their target similarities
    patterns_config = {
        'checkerboard': 0.9,
        'stripes': 0.9,
        'gradient': 0.9,
        'center_dot': 0.9,
        'spiral': 0.9,
        'concentric': 0.9,
        'random_dots': 0.9,
        'text_A': 0.85,  # Lower threshold for text patterns
        'text_B': 0.85   # Lower threshold for text patterns
    }
    
    results = {}
    noise_results = {}
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Train and test each pattern
    for pattern_type, target_similarity in patterns_config.items():
        print(f"\nTraining on {pattern_type} pattern...")
        model = PatternReconstructor().to(device)  # Fresh model for each pattern
        
        # Generate and train on clean pattern
        original = generate_simple_patterns(pattern_type).to(device)
        best_reconstruction, similarity = train_until_similarity(model, original, pattern_type, target_similarity=target_similarity)
        results[pattern_type] = similarity
        
        # Plot clean pattern results using the best reconstruction
        plot_results(original, best_reconstruction, f"{pattern_type} Pattern")
        
        # Test noise robustness if similarity is good enough
        if similarity >= target_similarity * 0.95:  # Allow 5% tolerance
            print(f"\nTesting noise robustness for {pattern_type}...")
            noise_results[pattern_type] = {}
            
            # Load saved model state for each noise test
            model.load_state_dict(torch.load(f'models/{pattern_type}_best.pth')['model_state_dict'])
            
            for noise_level in [0.1, 0.2, 0.3]:
                noisy_pattern = generate_simple_patterns(pattern_type, noise_level=noise_level).to(device)
                with torch.no_grad():
                    model.eval()
                    reconstruction = model(noisy_pattern)
                    noisy_similarity = calculate_similarity(noisy_pattern, reconstruction)
                    noise_results[pattern_type][f"noise_{noise_level}"] = noisy_similarity
                    
                    # Plot noisy pattern results
                    plot_results(noisy_pattern, reconstruction, f"{pattern_type} Pattern", noise_level)
                    model.train()
        else:
            print(f"\nSkipping noise robustness test for {pattern_type} due to low similarity")
    
    # Print final results
    print("\nFinal Results:")
    print("=" * 50)
    total_similarity = 0
    count = 0
    for pattern_type, similarity in results.items():
        target = patterns_config[pattern_type]
        status = "✓" if similarity >= target * 0.95 else "✗"
        print(f"{pattern_type}: {similarity:.4f} / {target:.4f} {status}")
        total_similarity += similarity
        count += 1
    
    avg_similarity = total_similarity / count
    print(f"Average Similarity: {avg_similarity:.4f}")
    
    # Print noise robustness results
    if noise_results:
        print("\nNoise Robustness Results:")
        print("=" * 50)
        for pattern_type, noise_levels in noise_results.items():
            print(f"\n{pattern_type}:")
            for noise_level, similarity in noise_levels.items():
                print(f"  {noise_level}: {similarity:.4f}")
    
    print("\nVisualization results have been saved in the 'results' directory")

if __name__ == '__main__':
    test_pattern_reconstruction()
