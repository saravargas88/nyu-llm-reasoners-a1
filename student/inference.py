
#INFERENCE TIME: now we use LM to predict next token and we decode it 
import torch
from student.Tokenizer import Tokenizer

import argparse

def arg_parse():
    p = argparse.ArgumentParser(description="Generate text from a trained Transformer LM")

    p.add_argument("--checkpoint",       type=str, required=True)
    p.add_argument("--tokenizer_vocab",  type=str, required=True)
    p.add_argument("--tokenizer_merges", type=str, required=True)
    p.add_argument("--vocab_size",     type=int,   default=10_000)
    p.add_argument("--context_length", type=int,   default=256)
    p.add_argument("--d_model",        type=int,   default=512)
    p.add_argument("--d_ff",           type=int,   default=1344)
    p.add_argument("--num_layers",     type=int,   default=4)
    p.add_argument("--num_heads",      type=int,   default=16)
    p.add_argument("--rope_theta",     type=float, default=10_000.0)
    p.add_argument("--prompt",         type=str)
    p.add_argument("--max_new_tokens", type=int,   default=256)
    p.add_argument("--temperature",    type=float, default=1.0)
    p.add_argument("--device",         type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    return p.parse_args()
    
def softmax_temp(logits, temp) -> torch.Tensor:
    # trick of subtracting max val in ith dim from all eelemtns for numerical stability
    scaled= logits/temp 

    scaled = scaled - scaled.max()
    exp_x = torch.exp(scaled)
    
    return exp_x / exp_x.sum()
    
def decoder(model , prompt_ids, max_tokens_generated, temperature,  device): 
    #sample tokens until hit end of text 
    model.eval()
    
    prompt= list(prompt_ids)
    generated= []
    
    for i in range(max_tokens_generated):
        
        context_tensor= torch.tensor([prompt], dtype= torch.long , device= device)
        
        #forward: infer over the vocab size 
        logits = model(context_tensor)
        
        #getr ontly last token 
        next_token_logits= logits[0, -1, :]
        
        probs= softmax_temp(logits= next_token_logits, temp= temperature)
        
        #sample token from distribution 
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        
        eos_id = tokenizer.token_to_id[b"<|endoftext|>"]
        if next_token_id == eos_id:
            break
        
        generated.append(next_token_id)
        prompt.append(next_token_id)
        
        
    return generated
        
from student.Transformer import TransformerLM 

        
if __name__ == "__main__":
    args = arg_parse()
    
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.tokenizer_vocab,
        merges_filepath=args.tokenizer_merges,
        special_tokens=["<|endoftext|>"],
    )
    
    #build model and get weights loaded 
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        theta=args.rope_theta,
    ).to(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"\nPrompt: {args.prompt}  ")
    print(f"temperature={args.temperature} \n")
    
    print("─" * 60)
    
    #generate 
    generated_ids = decoder(
        model=model,
        prompt_ids=prompt_ids,
        max_tokens_generated=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
    )
    
    generated_text = tokenizer.decode(generated_ids)
    print(args.prompt + generated_text)
    print(f"\nGenerated {len(generated_ids)} tokens.")
    



