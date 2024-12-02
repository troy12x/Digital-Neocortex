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
    criterion = nn.MSELoss()
    
    best_similarity = -float('inf')
    best_reconstruction = None
    best_model_state = None
    no_improve = 0
    min_improve = 1e-7
    
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
                # Combined loss
                mse_loss = mse_criterion(reconstruction, pattern)
                l1_loss = l1_criterion(reconstruction, pattern)
                loss = 0.7 * mse_loss + 0.3 * l1_loss
            
            if torch.isnan(loss):
                print("\nNaN loss detected, stopping training")
                break
                
            scaler.scale(loss).backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            
            with torch.no_grad():
                model.eval()
                reconstruction = model(pattern)
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
            else:
                no_improve += 1
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Similarity': f'{similarity:.4f}',
                'Best': f'{best_similarity:.4f}'
            })
            
            # Only stop if we reach target similarity
            if similarity >= target_similarity:
                print(f"\nReached target similarity of {target_similarity} at epoch {epoch+1}")
                return best_reconstruction, best_similarity
            
            # If stuck, try to escape local minimum
            if no_improve >= 300:
                no_improve = 0
                print(f"\nStuck at similarity {best_similarity:.4f}, attempting to escape local minimum...")
                # Increase learning rate temporarily
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate * 2
                    param_group['weight_decay'] *= 0.5
                print(f"Temporarily increased learning rate to {learning_rate * 2}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("\nWARNING: out of memory, clearing cache and retrying")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
        
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    
    print(f"\nReached max epochs. Best similarity: {best_similarity:.4f}")
    return best_reconstruction, best_similarity

def test_pattern_reconstruction():
    # Initialize model and move to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with gradient checkpointing disabled for testing
    model = PatternReconstructor().to(device)
    
    # Generate test patterns
    patterns = ['checkerboard', 'stripes', 'gradient', 'center_dot']
    
    fig, axes = plt.subplots(len(patterns), 2, figsize=(10, 15))
    fig.suptitle('Pattern Reconstruction Test')
    
    similarities = []
    pattern_results = []
    
    for i, pattern_type in enumerate(patterns):
        print(f"\nTraining on {pattern_type} pattern...")
        try:
            # Generate pattern
            original = generate_simple_patterns(pattern_type)
            
            # Train until high similarity
            reconstruction, similarity = train_until_similarity(model, original, pattern_type)
            
            if reconstruction is not None:
                # Move tensors to CPU for plotting
                original_cpu = original.cpu()
                reconstruction_cpu = reconstruction.cpu()
                
                # Plot
                axes[i, 0].imshow(original_cpu[0, 0].numpy(), cmap='viridis')
                axes[i, 0].set_title(f'Original {pattern_type}')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(reconstruction_cpu[0, 0].numpy(), cmap='viridis')
                axes[i, 1].set_title(f'Reconstruction\nSimilarity: {similarity:.4f}')
                axes[i, 1].axis('off')
                
                similarities.append(similarity)
                pattern_results.append((pattern_type, similarity))
            else:
                print(f"Failed to reconstruct {pattern_type} pattern")
                # Plot original only
                axes[i, 0].imshow(original.cpu()[0, 0].numpy(), cmap='viridis')
                axes[i, 0].set_title(f'Original {pattern_type}')
                axes[i, 0].axis('off')
                
                axes[i, 1].text(0.5, 0.5, 'Reconstruction Failed', 
                              horizontalalignment='center',
                              verticalalignment='center')
                axes[i, 1].axis('off')
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"WARNING: Skipping {pattern_type} due to memory constraints")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
            
        # Clear GPU memory
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    
    plt.tight_layout()
    
    # Print final summary
    print("\nFinal Results:")
    print("=" * 50)
    if pattern_results:
        for pattern, similarity in pattern_results:
            print(f"{pattern}: {similarity:.4f}")
        print(f"Average Similarity: {np.mean([s for _, s in pattern_results]):.4f}")
    else:
        print("No successful reconstructions")
    
    plt.show()

if __name__ == '__main__':
    test_pattern_reconstruction()
