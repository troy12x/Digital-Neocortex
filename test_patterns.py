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

def generate_simple_patterns(pattern_type='checkerboard', size=32, noise_level=0.0):
    """Generate patterns based on type"""
    pattern_functions = {
        'checkerboard': create_checkerboard_pattern,
        'stripes': create_stripes_pattern,
        'gradient': create_gradient_pattern,
        'center_dot': create_center_dot_pattern,
        'spiral': create_spiral_pattern,
        'concentric': create_concentric_circles_pattern,
        'random_dots': create_random_dots_pattern,
        'text_A': lambda size, noise_level: create_text_pattern(size, "A", noise_level),
        'text_B': lambda size, noise_level: create_text_pattern(size, "B", noise_level),
    }
    
    if pattern_type in pattern_functions:
        return pattern_functions[pattern_type](size=size, noise_level=noise_level)
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

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

def create_center_dot_pattern(size=32, noise_level=0.0):
    """Create a pattern with a distinct dot in the center"""
    pattern = torch.zeros((1, 1, size, size))
    center = size // 2
    
    # Create a more distinct center dot with a sharp edge
    dot_radius = size // 8  # Smaller, more distinct dot
    
    for i in range(size):
        for j in range(size):
            # Calculate distance from center
            dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            # Sharp edge for the dot
            if dist <= dot_radius:
                pattern[0, 0, i, j] = 1.0
            else:
                pattern[0, 0, i, j] = 0.0
    
    if noise_level > 0:
        pattern = add_noise(pattern, noise_level)
    return pattern
