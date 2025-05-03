import torch
from transformers import AutoTokenizer, AutoModel
from enhanced_digital_neocortex import DigitalNeocortex, NeocortexConfig
import torch.nn as nn

class BERTProcessor:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
    def get_embeddings(self, text_input: str) -> torch.Tensor:
        # Tokenize the input
        tokens = self.tokenizer(text_input, 
                              padding=True, 
                              truncation=True, 
                              max_length=512,
                              return_tensors="pt")
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state
            
        return embeddings

class InferenceNeocortex(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert_processor = BERTProcessor()
        self.neocortex = DigitalNeocortex(config)
        
    def forward(self, text_input: str) -> dict:
        # Get BERT embeddings
        embeddings = self.bert_processor.get_embeddings(text_input)
        
        # Process through Digital Neocortex
        outputs = self.neocortex(embeddings, modality='text')
        return outputs

def test_inference():
    # Test sentences
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "I love learning about artificial intelligence and neural networks.",
        "The weather is beautiful today.",
        "This is a test of the Digital Neocortex's processing capabilities."
    ]
    
    # Initialize configuration
    config = NeocortexConfig(
        input_dim=768,  # BERT hidden size
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        memory_size=100,  # Reduced for testing
        batch_size=1  # Processing one sentence at a time
    )
    
    # Initialize model
    model = InferenceNeocortex(config)
    
    # Process each test sentence
    print("\nTesting Digital Neocortex with BERT embeddings:")
    print("=" * 50)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\nTest {i}: Processing sentence:")
        print(f"Input: {sentence}")
        
        try:
            # Process the sentence
            outputs = model(sentence)
            
            # Print relevant outputs
            print("\nOutputs:")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: Shape {value.shape}")
                    # Print first few values
                    print(f"First few values: {value.flatten()[:5]}")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"Error processing sentence: {str(e)}")

if __name__ == "__main__":
    print("Starting inference test with BERT embeddings...")
    test_inference() 