import torch
import numpy as np
import cv2

def add_noise(pattern, noise_level=0.1):
    """Add Gaussian noise to the pattern"""
    noise = torch.randn_like(pattern) * noise_level
    noisy_pattern = pattern + noise
    return torch.clamp(noisy_pattern, 0, 1)

def create_checkerboard_pattern(size=32, noise_level=0.0):
    """Create a checkerboard pattern"""
    pattern = torch.zeros((1, 1, size, size))
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                pattern[0, 0, i, j] = 1.0
    if noise_level > 0:
        pattern = add_noise(pattern, noise_level)
    return pattern

def create_stripes_pattern(size=32, noise_level=0.0):
    """Create a striped pattern"""
    pattern = torch.zeros((1, 1, size, size))
    for i in range(size):
        if i % 4 < 2:
            pattern[0, 0, i, :] = 1.0
    if noise_level > 0:
        pattern = add_noise(pattern, noise_level)
    return pattern

def create_gradient_pattern(size=32, noise_level=0.0):
    """Create a gradient pattern"""
    x = torch.linspace(0, 1, size)
    y = torch.linspace(0, 1, size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    pattern = (xx + yy) / 2
    pattern = pattern.unsqueeze(0).unsqueeze(0)
    if noise_level > 0:
        pattern = add_noise(pattern, noise_level)
    return pattern

def create_center_dot_pattern(size=32, noise_level=0.0):
    """Create a pattern with a dot in the center"""
    pattern = torch.zeros((1, 1, size, size))
    center = size // 2
    radius = size // 8
    for i in range(size):
        for j in range(size):
            if (i - center) ** 2 + (j - center) ** 2 < radius ** 2:
                pattern[0, 0, i, j] = 1.0
    if noise_level > 0:
        pattern = add_noise(pattern, noise_level)
    return pattern

def create_spiral_pattern(size=32, noise_level=0.0):
    """Create a spiral pattern"""
    pattern = torch.zeros((1, 1, size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            r = np.sqrt(x*x + y*y)
            theta = np.arctan2(y, x)
            value = (r + 4*theta) % (size/2)
            pattern[0, 0, i, j] = 1.0 if value < size/4 else 0.0
    if noise_level > 0:
        pattern = add_noise(pattern, noise_level)
    return pattern

def create_concentric_circles_pattern(size=32, noise_level=0.0):
    """Create concentric circles pattern"""
    pattern = torch.zeros((1, 1, size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            r = np.sqrt((i - center)**2 + (j - center)**2)
            pattern[0, 0, i, j] = 1.0 if int(r) % 4 < 2 else 0.0
    if noise_level > 0:
        pattern = add_noise(pattern, noise_level)
    return pattern

def create_random_dots_pattern(size=32, num_dots=10, noise_level=0.0):
    """Create random dots pattern"""
    pattern = torch.zeros((1, 1, size, size))
    for _ in range(num_dots):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        radius = np.random.randint(1, 4)
        for i in range(max(0, x-radius), min(size, x+radius+1)):
            for j in range(max(0, y-radius), min(size, y+radius+1)):
                if (i-x)**2 + (j-y)**2 <= radius**2:
                    pattern[0, 0, i, j] = 1.0
    if noise_level > 0:
        pattern = add_noise(pattern, noise_level)
    return pattern

def create_text_pattern(size=32, text="A", noise_level=0.0):
    """Create a pattern with text"""
    pattern = torch.zeros((1, 1, size, size))
    img = np.zeros((size, size), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = size/32
    thickness = max(1, int(size/32))
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = (size - text_size[0]) // 2
    y = (size + text_size[1]) // 2
    cv2.putText(img, text, (x, y), font, font_scale, 255, thickness)
    pattern[0, 0] = torch.from_numpy(img).float() / 255.0
    if noise_level > 0:
        pattern = add_noise(pattern, noise_level)
    return pattern

def create_noise_pattern(size=32, base_pattern_func=create_checkerboard_pattern, noise_levels=[0.1, 0.2, 0.3, 0.4]):
    """Create multiple versions of a pattern with different noise levels"""
    patterns = []
    for noise_level in noise_levels:
        pattern = base_pattern_func(size=size, noise_level=noise_level)
        patterns.append(pattern)
    return patterns

def generate_simple_patterns(pattern_type, size=64, noise_level=0.0):
    """Generate simple geometric patterns."""
    pattern = torch.zeros((1, 1, size, size))
    center = size // 2
    radius = size // 4
    
    if pattern_type == "circle":
        for i in range(size):
            for j in range(size):
                if (i - center) ** 2 + (j - center) ** 2 <= radius ** 2:
                    pattern[0, 0, i, j] = 1.0
                    
    elif pattern_type == "square":
        start = center - radius
        end = center + radius
        pattern[0, 0, start:end, start:end] = 1.0
        
    elif pattern_type == "triangle":
        # Generate an equilateral triangle
        height = int(radius * 1.732)  # sqrt(3) ≈ 1.732
        
        # Calculate triangle vertices
        x0, y0 = center, center - height//2  # top vertex
        x1, y1 = center - radius, center + height//2  # bottom left
        x2, y2 = center + radius, center + height//2  # bottom right
        
        # Fill triangle using barycentric coordinates
        for i in range(center - radius, center + radius + 1):
            for j in range(center - height//2, center + height//2 + 1):
                # Calculate barycentric coordinates
                w1 = ((y2 - y1) * (i - x1) + (x1 - x2) * (j - y1)) / float((y2 - y1) * (x0 - x1) + (x1 - x2) * (y0 - y1))
                w2 = ((y0 - y2) * (i - x2) + (x2 - x0) * (j - y2)) / float((y0 - y2) * (x1 - x2) + (x2 - x0) * (y1 - y2))
                w0 = 1 - w1 - w2
                
                # If point is inside triangle
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    pattern[0, 0, j, i] = 1.0
        
    elif pattern_type == "hexagon":
        # Generate regular hexagon
        for i in range(size):
            for j in range(size):
                x = i - center
                y = j - center
                if abs(x) <= radius and abs(y) <= radius * 0.866:  # sqrt(3)/2 ≈ 0.866
                    if abs(y) <= radius * 0.866 - (radius * 0.866 / radius) * abs(x):
                        pattern[0, 0, i, j] = 1.0
                        
    elif pattern_type == "spiral":
        # Generate spiral pattern
        for t in np.linspace(0, 6*np.pi, 1000):
            r = t * radius / (6*np.pi)
            x = int(center + r * np.cos(t))
            y = int(center + r * np.sin(t))
            if 0 <= x < size and 0 <= y < size:
                pattern[0, 0, x, y] = 1.0
                
        # Thicken the spiral
        kernel_size = 3
        kernel = torch.ones((1, 1, kernel_size, kernel_size))
        pattern = torch.nn.functional.conv2d(
            pattern, 
            kernel, 
            padding=kernel_size//2
        )
        pattern = (pattern > 0).float()
    
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    # Add noise if specified
    if noise_level > 0:
        noise = torch.randn_like(pattern) * noise_level
        pattern = torch.clamp(pattern + noise, 0, 1)
    
    return pattern

def test_noise_robustness(model, pattern_type='checkerboard', noise_levels=[0.1, 0.2, 0.3]):
    """Test model's robustness to noise"""
    results = {}
    for noise_level in noise_levels:
        pattern = generate_simple_patterns(pattern_type, noise_level=noise_level)
        with torch.no_grad():
            reconstruction = model(pattern)
            similarity = calculate_similarity(pattern, reconstruction)
            results[f"noise_{noise_level}"] = similarity
    return results

def calculate_similarity(pattern1, pattern2):
    """Calculate similarity between two patterns"""
    # TO DO: implement similarity calculation
    pass

def test_pattern_batch(batch_size=32, size=32):
    """Generate a batch of test patterns"""
    patterns = []
    pattern_types = ['checkerboard', 'stripes', 'gradient', 'center_dot']
    
    for _ in range(batch_size):
        pattern_type = pattern_types[_ % len(pattern_types)]
        patterns.append(generate_simple_patterns(pattern_type, size))
        
    return torch.cat(patterns, dim=0)

def generate_test_patterns(batch_size=32, size=64):
    """Generate a diverse set of test patterns for evaluation."""
    patterns = []
    pattern_types = [
        'checkerboard',
        'stripes',
        'gradient',
        'center_dot',
        'spiral',
        'concentric',
        'random_dots',
        'text'
    ]
    
    for _ in range(batch_size):
        # Randomly select pattern type
        pattern_type = np.random.choice(pattern_types)
        
        # Generate base pattern
        if pattern_type == 'checkerboard':
            pattern = create_checkerboard_pattern(size)
        elif pattern_type == 'stripes':
            pattern = create_stripes_pattern(size)
        elif pattern_type == 'gradient':
            pattern = create_gradient_pattern(size)
        elif pattern_type == 'center_dot':
            pattern = create_center_dot_pattern(size)
        elif pattern_type == 'spiral':
            pattern = create_spiral_pattern(size)
        elif pattern_type == 'concentric':
            pattern = create_concentric_circles_pattern(size)
        elif pattern_type == 'random_dots':
            pattern = create_random_dots_pattern(size)
        else:  # text
            pattern = create_text_pattern(size, text=chr(np.random.randint(65, 91)))
        
        # Convert to complex representation (2 channels)
        magnitude = pattern.squeeze()
        phase = torch.rand_like(magnitude) * 2 * np.pi
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        complex_pattern = torch.stack([real, imag], dim=0).unsqueeze(0)
        
        # Normalize each channel independently
        for c in range(2):
            channel = complex_pattern[:, c:c+1]
            min_val = channel.min()
            max_val = channel.max()
            if max_val > min_val:
                complex_pattern[:, c:c+1] = (channel - min_val) / (max_val - min_val)
        
        patterns.append(complex_pattern)
    
    # Stack into batch
    batch = torch.cat(patterns, dim=0)
    
    # Verify tensor properties
    assert not torch.isnan(batch).any(), "NaN values in generated patterns"
    assert not torch.isinf(batch).any(), "Inf values in generated patterns"
    assert batch.min() >= 0 and batch.max() <= 1, "Values outside [0,1] range"
    assert batch.size(1) == 2, "Wrong number of channels"
    assert batch.size(-1) == size and batch.size(-2) == size, "Wrong spatial dimensions"
    
    return batch

def add_noise_to_patterns(patterns: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
    """Add controlled noise to patterns for robustness testing."""
    noise = torch.randn_like(patterns) * noise_level
    return patterns + noise

def create_corrupted_patterns(patterns: torch.Tensor, corruption_ratio: float = 0.3):
    """Create partially corrupted patterns for reconstruction testing."""
    # Create binary mask (1 = keep, 0 = corrupt)
    mask = torch.bernoulli(torch.full_like(patterns, 1 - corruption_ratio))
    
    # Apply mask and add noise to corrupted regions
    corrupted = patterns * mask
    noise = torch.randn_like(patterns) * 0.1
    corrupted = corrupted + (1 - mask) * noise
    
    # Normalize corrupted patterns
    for b in range(corrupted.size(0)):
        for c in range(corrupted.size(1)):
            channel = corrupted[b, c:c+1]
            min_val = channel.min()
            max_val = channel.max()
            if max_val > min_val:
                corrupted[b, c:c+1] = (channel - min_val) / (max_val - min_val)
    
    return corrupted

def generate_pattern_sequence(length: int = 10, size: int = 64) -> torch.Tensor:
    """Generate a sequence of related patterns for temporal pattern testing."""
    sequence = []
    base_pattern = generate_test_patterns(batch_size=1, size=size)
    
    for i in range(length):
        # Evolve the pattern
        evolved = base_pattern + torch.randn_like(base_pattern) * 0.1 * i
        sequence.append(evolved)
    
    return torch.cat(sequence, dim=0)
