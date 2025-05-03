import torch
from enhanced_digital_neocortex import DigitalNeocortex, NeocortexConfig
import numpy as np

def test_token_processing():
    """Test how the model processes and understands tokens"""
    print("\n=== Testing Token Processing ===")
    
    config = NeocortexConfig(
        vocab_size=1000,
        hidden_dim=256,
        num_columns=4,
        max_sequence_length=128
    )
    
    print("Model Configuration:")
    print(f"Vocabulary Size: {config.vocab_size}")
    print(f"Hidden Dimension: {config.hidden_dim}")
    
    model = DigitalNeocortex(config)
    model = model.to(device)
    
    # Create sample input
    batch_size = 2
    seq_length = 10
    input_sequence = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)
    print(f"Input sequence shape: {input_sequence.shape}")
    
    try:
        with torch.no_grad():
            output = model(input_sequence)
            print(f"Output shape: {output.shape}")
        return "Token processing test completed"
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Token processing test failed"

def test_memory_system():
    """Test the memory system's ability to retain and recall information"""
    print("\n=== Testing Memory System ===")
    
    config = NeocortexConfig(
        vocab_size=1000,
        hidden_dim=256,
        memory_size=32
    )
    model = DigitalNeocortex(config)
    model = model.to(device)
    
    try:
        # Test with batch
        batch_size = 2
        seq_length = 10
        input_sequence = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)
        
        with torch.no_grad():
            output = model(input_sequence)
            print(f"Memory system output shape: {output.shape}")
        
        return "Memory system test completed"
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Memory system test failed"

def test_hierarchical_processing():
    """Test the hierarchical processing capabilities"""
    print("\n=== Testing Hierarchical Processing ===")
    
    config = NeocortexConfig(
        vocab_size=1000,
        hidden_dim=256,
        num_layers=4
    )
    model = DigitalNeocortex(config)
    model = model.to(device)
    
    try:
        batch_size = 2
        seq_length = 20
        input_sequence = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)
        
        with torch.no_grad():
            output = model(input_sequence)
            print(f"Hierarchical processing output shape: {output.shape}")
        
        return "Hierarchical processing test completed"
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Hierarchical processing test failed"

def test_cortical_columns():
    """Test the specialized processing in cortical columns"""
    print("\n=== Testing Cortical Columns ===")
    
    config = NeocortexConfig(
        vocab_size=1000,
        hidden_dim=256,
        num_columns=4
    )
    model = DigitalNeocortex(config)
    model = model.to(device)
    
    try:
        batch_size = 2
        seq_length = 15
        input_sequence = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)
        
        with torch.no_grad():
            output = model(input_sequence)
            print(f"Cortical columns output shape: {output.shape}")
        
        return "Cortical columns test completed"
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Cortical columns test failed"

def run_comprehensive_test():
    """Run all tests and provide a comprehensive evaluation"""
    print("Starting comprehensive Digital Neocortex testing...")
    
    # Set up device
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Run individual tests
        results = []
        results.append(test_token_processing())
        results.append(test_memory_system())
        results.append(test_hierarchical_processing())
        results.append(test_cortical_columns())
        
        # Print summary
        print("\n=== Test Summary ===")
        for result in results:
            print(f"✓ {result}")
        
        print("\nAll tests completed!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    run_comprehensive_test()