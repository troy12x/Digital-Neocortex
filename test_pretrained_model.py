import torch
import argparse
import os
from train_digital_neocortex import DigitalNeocortexModel, DigitalNeocortexTokenizer

def load_pretrained_model(model_path, tokenizer_path, device='cpu'):
    """
    Load pretrained model and tokenizer.
    """
    print(f"Loading model from {model_path}")
    model = DigitalNeocortexModel.load(model_path, device)
    model.to(device)
    
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = DigitalNeocortexTokenizer.load(tokenizer_path)
    
    return model, tokenizer

def predict_next_token(model, tokenizer, prompt, device='cpu'):
    """
    Predict the next token for a given prompt.
    """
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, max_length=20)
    input_tensor = torch.LongTensor([input_ids]).to(device)
    
    # Reset model state
    model.reset_state()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # Get next token prediction
    next_token_logits = outputs[0, -1, :]
    
    # Get top 5 predictions
    top_k = 5
    topk_values, topk_indices = torch.topk(next_token_logits, top_k)
    
    # Convert to probabilities
    topk_probs = torch.softmax(topk_values, dim=0)
    
    # Get predictions
    predictions = []
    for i in range(top_k):
        token_id = topk_indices[i].item()
        token = tokenizer.idx_to_word.get(token_id, "<unknown>")
        prob = topk_probs[i].item()
        predictions.append((token, prob))
    
    return predictions

def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, repetition_penalty=1.2, device='cpu'):
    """
    Generate text from a prompt.
    """
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, max_length=20)
    
    # Reset model state
    model.reset_state()
    
    # Generate tokens
    generated = input_ids.copy()
    
    for _ in range(max_length - len(input_ids)):
        # Create input tensor
        input_tensor = torch.LongTensor([generated[-20:]]).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Get next token prediction
        next_token_logits = outputs[0, -1, :]
        
        # Apply repetition penalty
        for token_id in set(generated):
            next_token_logits[token_id] /= repetition_penalty
        
        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        
        # Sample from the distribution
        probs = torch.softmax(next_token_logits, dim=0)
        
        # Prevent sampling tokens that are already in the prompt
        if len(generated) <= len(input_ids):
            # If we're still in the original prompt length, avoid repeating
            for i in range(len(probs)):
                if i in input_ids:
                    probs[i] = 0.0
            
            # Renormalize
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                # If all probabilities were zeroed, reset
                probs = torch.ones_like(probs) / len(probs)
        
        # Sample
        next_token = torch.multinomial(probs, 1).item()
        
        # Stop if end of sequence
        if next_token == tokenizer.special_tokens['<eos>']:
            break
        
        # Add to generated sequence
        generated.append(next_token)
    
    # Decode generated sequence
    generated_text = tokenizer.decode(generated)
    
    return generated_text

def interactive_mode(model, tokenizer, device='cpu'):
    """
    Interactive mode for text generation.
    """
    print("\n===== Digital Neocortex Interactive Mode =====")
    print("Type 'exit' to quit, 'help' for commands")
    
    temperature = 1.0
    repetition_penalty = 1.2
    
    while True:
        try:
            user_input = input("\nEnter prompt: ")
            
            if user_input.lower() == 'exit':
                break
            
            if user_input.lower() == 'help':
                print("\nCommands:")
                print("  'exit' - Exit interactive mode")
                print("  'help' - Show this help message")
                print("  'next <prompt>' - Predict next token for prompt")
                print("  'gen <prompt>' - Generate text from prompt")
                print("  'temp <value>' - Set temperature (default: 1.0)")
                print("  'penalty <value>' - Set repetition penalty (default: 1.2)")
                continue
            
            if user_input.lower().startswith('next '):
                prompt = user_input[5:]
                predictions = predict_next_token(model, tokenizer, prompt, device)
                
                print("\nNext token predictions:")
                for i, (token, prob) in enumerate(predictions):
                    print(f"  {i+1}. '{token}' ({prob:.4f})")
                continue
            
            if user_input.lower().startswith('temp '):
                try:
                    temperature = float(user_input[5:])
                    print(f"Temperature set to {temperature}")
                except ValueError:
                    print("Invalid temperature value. Please enter a number.")
                continue
            
            if user_input.lower().startswith('penalty '):
                try:
                    repetition_penalty = float(user_input[8:])
                    print(f"Repetition penalty set to {repetition_penalty}")
                except ValueError:
                    print("Invalid repetition penalty value. Please enter a number.")
                continue
            
            # Default: generate text
            if user_input.lower().startswith('gen '):
                user_input = user_input[4:]
            
            generated_text = generate_text(model, tokenizer, user_input, 
                                          temperature=temperature, 
                                          repetition_penalty=repetition_penalty, 
                                          device=device)
            print("\nGenerated text:")
            print(generated_text)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nExiting interactive mode.")

def main():
    parser = argparse.ArgumentParser(description='Test pretrained Digital Neocortex model')
    parser.add_argument('--model_path', type=str, default='pretrained/digital_neocortex_model.pt', help='Path to pretrained model')
    parser.add_argument('--tokenizer_path', type=str, default='pretrained/digital_neocortex_tokenizer.json', help='Path to tokenizer')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--prompt', type=str, default=None, help='Prompt for text generation')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for text generation')
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help='Repetition penalty for text generation')
    
    args = parser.parse_args()
    
    # Check if model and tokenizer exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    if not os.path.exists(args.tokenizer_path):
        print(f"Error: Tokenizer file not found at {args.tokenizer_path}")
        return
    
    # Load model and tokenizer
    model, tokenizer = load_pretrained_model(args.model_path, args.tokenizer_path, args.device)
    
    # Interactive mode or single prompt
    if args.interactive:
        interactive_mode(model, tokenizer, args.device)
    elif args.prompt:
        generated_text = generate_text(model, tokenizer, args.prompt, 
                                      temperature=args.temperature, 
                                      repetition_penalty=args.repetition_penalty, 
                                      device=args.device)
        print("\nGenerated text:")
        print(generated_text)
    else:
        # Default prompts to test
        test_prompts = [
            "what is the capital of",
            "how many people live in",
            "what is the largest",
            "who invented the"
        ]
        
        print("\nTesting with default prompts:")
        for prompt in test_prompts:
            print(f"\nPrompt: '{prompt}'")
            
            # Predict next token
            predictions = predict_next_token(model, tokenizer, prompt, args.device)
            print("Next token predictions:")
            for i, (token, prob) in enumerate(predictions):
                print(f"  {i+1}. '{token}' ({prob:.4f})")
            
            # Generate text
            generated_text = generate_text(model, tokenizer, prompt, 
                                          temperature=args.temperature, 
                                          repetition_penalty=args.repetition_penalty, 
                                          device=args.device)
            print("Generated text:")
            print(generated_text)

if __name__ == "__main__":
    main()
