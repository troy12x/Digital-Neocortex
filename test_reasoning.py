import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from test_patterns import generate_simple_patterns
from test_reconstruction import PatternReconstructor, calculate_similarity
import traceback

class NeocortexReasoning(nn.Module):
    def __init__(self, base_features=64):
        super(NeocortexReasoning, self).__init__()
        
        # Text embedding for cross-modality reasoning
        self.text_embedding = nn.Sequential(
            nn.Linear(768, base_features * 4),  # 768 is BERT embedding size
            nn.ReLU(),
            nn.Linear(base_features * 4, base_features * 4)
        )
        
        # Enhanced reasoning encoder with attention
        self.reasoning_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, base_features, 3, padding=1),
                nn.LayerNorm([base_features, 64, 64]),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(base_features, base_features*2, 4, stride=2, padding=1),
                nn.LayerNorm([base_features*2, 32, 32]),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(base_features*2, base_features*4, 4, stride=2, padding=1),
                nn.LayerNorm([base_features*4, 16, 16]),
                nn.LeakyReLU(0.2)
            )
        ])
        
        # Self-attention mechanism
        self.self_attention = nn.MultiheadAttention(
            embed_dim=base_features*4,
            num_heads=8,
            batch_first=True
        )
        
        # Pattern relationship understanding with residual connections
        self.relationship_processor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_features*4, base_features*4, 3, padding=1),
                nn.LayerNorm([base_features*4, 16, 16]),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(base_features*4, base_features*4, 3, padding=1),
                nn.LayerNorm([base_features*4, 16, 16]),
                nn.LeakyReLU(0.2)
            )
        ])
        
        # Enhanced decoder with skip connections
        self.pattern_predictor = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(base_features*4, base_features*2, 4, stride=2, padding=1),
                nn.LayerNorm([base_features*2, 32, 32]),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(base_features*2, base_features, 4, stride=2, padding=1),
                nn.LayerNorm([base_features, 64, 64]),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(base_features, 1, 3, padding=1),
                nn.Sigmoid()
            )
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.LayerNorm)):
            if hasattr(m, 'weight'):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, pattern1, pattern2, text_embedding=None):
        # Ensure inputs are proper 4D tensors [batch, channel, height, width]
        if len(pattern1.shape) == 2:
            pattern1 = pattern1.unsqueeze(0).unsqueeze(0)
        elif len(pattern1.shape) == 3:
            pattern1 = pattern1.unsqueeze(1)
            
        if len(pattern2.shape) == 2:
            pattern2 = pattern2.unsqueeze(0).unsqueeze(0)
        elif len(pattern2.shape) == 3:
            pattern2 = pattern2.unsqueeze(1)
        
        # Combine patterns
        x = torch.cat([pattern1, pattern2], dim=1)
        
        # Encoding with skip connections
        skip_connections = []
        for encoder in self.reasoning_encoder:
            x = encoder(x)
            skip_connections.append(x.clone())  # Clone to preserve size
        
        # Apply self-attention
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_att, _ = self.self_attention(x_flat, x_flat, x_flat)
        x = x_att.transpose(1, 2).view(b, c, h, w)
        
        # Incorporate text embedding if provided
        if text_embedding is not None:
            text_features = self.text_embedding(text_embedding)
            text_features = text_features.view(b, -1, 1, 1).expand(-1, -1, h, w)
            x = x + text_features
        
        # Process relationships with residual connections
        for processor in self.relationship_processor:
            residual = x
            x = processor(x)
            x = x + residual
        
        # Decoding with skip connections
        for i, decoder in enumerate(self.pattern_predictor[:-1]):
            x = decoder(x)
            if i < len(skip_connections):
                skip = skip_connections[-(i+1)]
                
                # Match spatial dimensions
                if x.shape[2:] != skip.shape[2:]:
                    skip = nn.functional.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
                
                # Match channel dimensions using 1x1 convolution
                channel_matcher = nn.Conv2d(skip.shape[1], x.shape[1], kernel_size=1).to(x.device)
                skip = channel_matcher(skip)
                
                x = x + skip
        
        # Final prediction
        x = self.pattern_predictor[-1](x)
        return x.squeeze(1)

def add_noise(pattern, noise_level=0.1):
    """Add random noise to a pattern."""
    noise = torch.randn_like(pattern) * noise_level
    noisy_pattern = pattern + noise
    return torch.clamp(noisy_pattern, 0, 1)

def generate_complex_pattern(size=64, pattern_type='transition'):
    """Generate complex geometric patterns with transitions."""
    pattern = torch.zeros((1, size, size))
    center = size // 2
    if pattern_type == 'square':
        side = size // 4
        pattern[0, center-side:center+side, center-side:center+side] = 1
    elif pattern_type == 'circle':
        for i in range(size):
            for j in range(size):
                if (i - center) ** 2 + (j - center) ** 2 <= (size//4) ** 2:
                    pattern[0, i, j] = 1
    elif pattern_type == 'hexagon':
        import math
        radius = size // 4
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                if abs(x) <= radius * math.sqrt(3)/2 and abs(y) <= radius:
                    pattern[0, i, j] = 1
    return pattern

def generate_reasoning_sequence(pattern_type, sequence_length=3):
    """Generate a sequence of patterns for reasoning tasks."""
    patterns = []
    
    if pattern_type == "rotation":
        # Create a rotating spiral pattern
        base_pattern = generate_simple_patterns("spiral", size=64, noise_level=0)[0, 0]
        for i in range(sequence_length):
            angle = i * 45  # Rotate by 45 degrees each step
            
            # Create rotation matrix
            theta = np.deg2rad(angle)
            rot_matrix = torch.tensor([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ], dtype=torch.float32)
            
            # Create coordinate grid
            x = torch.linspace(-1, 1, 64)
            y = torch.linspace(-1, 1, 64)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
            
            # Apply rotation
            rotated_coords = torch.mm(coords, rot_matrix)
            grid_x = rotated_coords[:, 0].view(64, 64)
            grid_y = rotated_coords[:, 1].view(64, 64)
            
            # Sample from original pattern
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
            pattern = torch.nn.functional.grid_sample(
                base_pattern.unsqueeze(0).unsqueeze(0),
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            ).squeeze()
            patterns.append(pattern)
    
    elif pattern_type == "scale":
        # Create a scaling circle pattern
        base_pattern = generate_simple_patterns("circle", size=64, noise_level=0)[0, 0]
        for i in range(sequence_length):
            scale = 1.0 + i * 0.5  # Increase size by 50% each step
            
            # Create coordinate grid
            x = torch.linspace(-1, 1, 64)
            y = torch.linspace(-1, 1, 64)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            
            # Apply scaling
            grid_x = grid_x / scale
            grid_y = grid_y / scale
            
            # Sample from original pattern
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
            pattern = torch.nn.functional.grid_sample(
                base_pattern.unsqueeze(0).unsqueeze(0),
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            ).squeeze()
            patterns.append(pattern)
    
    elif pattern_type == "shape_transition":
        # Create shape transition sequence
        shapes = ["square", "hexagon", "circle"]
        for shape_name in shapes[:sequence_length]:
            pattern = generate_simple_patterns(shape_name, size=64, noise_level=0)[0, 0]
            patterns.append(pattern)
   
    elif pattern_type == "composite":
        # Create a pattern that combines rotation and scaling
        base_pattern = generate_simple_patterns("triangle", size=64, noise_level=0)[0, 0]
        for i in range(sequence_length):
            angle = i * 30  # Rotate by 30 degrees each step
            scale = 1.0 + i * 0.3  # Increase size by 30% each step
            
            # Create rotation matrix
            theta = np.deg2rad(angle)
            rot_matrix = torch.tensor([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ], dtype=torch.float32)
            
            # Create coordinate grid
            x = torch.linspace(-1, 1, 64)
            y = torch.linspace(-1, 1, 64)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
            
            # Apply rotation
            rotated_coords = torch.mm(coords, rot_matrix)
            grid_x = rotated_coords[:, 0].view(64, 64)
            grid_y = rotated_coords[:, 1].view(64, 64)
            
            # Apply scaling
            grid_x = grid_x / scale
            grid_y = grid_y / scale
            
            # Sample from original pattern
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
            pattern = torch.nn.functional.grid_sample(
                base_pattern.unsqueeze(0).unsqueeze(0),
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            ).squeeze()
            patterns.append(pattern)
    
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    return torch.stack(patterns)

def get_text_embedding(text_description):
    """
    Generate text embedding using pre-trained BERT model with caching.
    
    Args:
        text_description (str): Input text description
    
    Returns:
        torch.Tensor: Extracted text embedding
    """
    if not text_description:
        return None
        
    # Use cache if available
    if hasattr(get_text_embedding, 'cache'):
        if text_description in get_text_embedding.cache:
            return get_text_embedding.cache[text_description]
    else:
        get_text_embedding.cache = {}
    
    try:
        # Ensure the BERT model is in evaluation mode
        bert_model.eval()
        
        # Tokenize the input text
        inputs = tokenizer(
            text_description, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # Disable gradient computation for embedding extraction
        with torch.no_grad():
            # Move inputs to the same device as the model
            inputs = {k: v.to(bert_model.device) for k, v in inputs.items()}
            
            # Extract embeddings
            outputs = bert_model(**inputs)
            
            # Use the [CLS] token embedding (first token)
            text_embedding = outputs.last_hidden_state[:, 0, :]
            
            # Cache the result
            get_text_embedding.cache[text_description] = text_embedding
            
            return text_embedding
            
    except Exception as e:
        print(f"Warning: Error generating text embedding: {str(e)}")
        return None

def train_reasoning(model, pattern_type, max_epochs=5000, learning_rate=0.001, text_description=None):
    """Enhanced training with text-guided pattern generation."""
    # Disable anomaly detection to prevent performance overhead
    torch.autograd.set_detect_anomaly(False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Generate pattern sequence
    patterns = generate_reasoning_sequence(pattern_type, sequence_length=3)
    patterns = patterns.to(device)
    
    # Get text embedding if provided
    text_embedding = None
    if text_description:
        text_embedding = get_text_embedding(text_description)
        if text_embedding is not None:
            text_embedding = text_embedding.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_loss = float('inf')
    best_prediction = None
    
    pbar = tqdm(range(max_epochs), desc=f"Training {pattern_type} reasoning")
    for epoch in pbar:
        # Zero out gradients at the start of each epoch
        optimizer.zero_grad(set_to_none=True)
        
        # Create fresh tensors for each iteration
        input1 = patterns[0].clone()
        input2 = patterns[1].clone()
        target = patterns[2].clone()
        
        # Ensure all tensors are on the correct device
        input1 = input1.to(device)
        input2 = input2.to(device)
        target = target.to(device)
        
        # Forward pass
        model.train()
        prediction = model(input1, input2, text_embedding)
        
        # Ensure prediction matches target shape
        if prediction.shape != target.shape:
            prediction = prediction.view_as(target)
        
        # Compute loss
        loss = criterion(prediction, target)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Update best loss and prediction
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_prediction = prediction.detach().cpu()
        
        # Update progress bar
        pbar.set_postfix({'Loss': f"{current_loss:.4f}", 'Best': f"{best_loss:.4f}"})
        
        # Early stopping
        if best_loss < 0.01 or (epoch > 1000 and current_loss > 1.0):
            print(f"Converged at epoch {epoch}")
            break
    
    print(f"Best loss for {pattern_type}: {best_loss:.4f}")
    return best_prediction, best_loss

def plot_reasoning_results(original_sequence, predicted, pattern_type, loss):
    """Plot the original sequence and predicted pattern."""
    plt.figure(figsize=(15, 5))
    
    # Plot original sequence
    for i in range(3):
        plt.subplot(1, 4, i + 1)
        pattern = original_sequence[i]
        # Remove batch dimension if present and ensure 2D
        if len(pattern.shape) == 3:
            pattern = pattern.squeeze(0)
        plt.imshow(pattern.cpu().numpy(), cmap='gray')
        plt.title(f'Original {i+1}')
        plt.axis('off')
    
    # Plot predicted pattern
    plt.subplot(1, 4, 4)
    # Remove batch dimension if present and ensure 2D
    if len(predicted.shape) == 3:
        predicted = predicted.squeeze(0)
    plt.imshow(predicted.cpu().numpy(), cmap='gray')
    plt.title(f'Predicted\nLoss: {loss:.4f}')
    plt.axis('off')
    
    plt.suptitle(f'Pattern Reasoning Results: {pattern_type}')
    plt.tight_layout()
    plt.show()

def test_pattern_reasoning():
    """Test the model's ability to understand and predict pattern transformations"""
    # Initialize model
    model = NeocortexReasoning()
    
    # Test composite transformation (rotation + scaling)
    test_case = ("composite", "Rotate the triangle while gradually increasing its size")
    pattern_type, description = test_case
    
    print(f"\nTesting {pattern_type} reasoning...")
    print(f"Description: {description}")
    
    try:
        # Train model and get predictions
        predicted, loss = train_reasoning(model, pattern_type, text_description=description)
        
        # Generate original sequence for visualization
        original_sequence = generate_reasoning_sequence(pattern_type)
        
        # Plot results
        plot_reasoning_results(original_sequence, predicted, pattern_type, loss)
        
    except Exception as e:
        print(f"Error testing {pattern_type}: {str(e)}")
        traceback.print_exc()  # Print full traceback for debugging

if __name__ == "__main__":
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    test_pattern_reasoning()
