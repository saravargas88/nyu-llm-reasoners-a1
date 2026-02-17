

from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import time


# training code: 
def cross_entropy(logits, targets):
    max_logits = logits.max(dim=-1, keepdim=True).values  

    shifted = logits - max_logits 

    # denominator term
    log_sum_exp = shifted.exp().sum(dim=-1).log()

    # logit at target, shifted for numerical stability
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    numerically_stable_logits = target_logits - max_logits.squeeze(-1)

    loss = -numerically_stable_logits + log_sum_exp

    return loss.mean()



class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        
        
    def step(self, closure: Optional[Callable] = None): 
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
                
            for p in group["params"]: 
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value. grad = p.grad.data # Get the gradient of loss with respect to p.
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place. state["t"] = t + 1 # Increment iteration number.
                state["t"] = t + 1 # Increment iteration number.
            
            
        return loss
    
    
    

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999) , eps=  1e-8, weight_decay= 0.01):
        

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps= group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad  = p.grad.data
                state = self.state[p]

                # initialise state on first step
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m, v = state["m"], state["v"]
                state["t"] += 1
                t= state["t"]

                
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                
                alpha_t = lr * math.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)

               
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

               
                p.data.mul_(1.0 - lr * weight_decay)

        return loss

    
def learning_rate_scheduler(t, lr_max, lr_min, Tw, Tc):
    
    import math
    if t < Tw:
         
        return (t / Tw) * lr_max
    elif t <= Tc:
        
        progress = (t - Tw) / (Tc - Tw)         
        return lr_min + 0.5 * (1 + math.cos(math.pi * progress)) * (lr_max - lr_min)
    else:
        return lr_min


def gradient_clipping(parameters, M: float, eps: float = 1e-6):
    #M is the maximum norm value
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return

    # l2 norm all params
    l2_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads))

    if l2_norm >M:
        scale = M / (l2_norm + eps)
        for g in grads:
            g.mul_(scale)   
 
#data loading
import numpy as np
def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    ix = np.random.randint(0, len(x) - context_length, size=batch_size)
    inputs  = np.stack([x[i : i + context_length    ] for i in ix])
    targets = np.stack([x[i + 1 : i + context_length + 1] for i in ix])
    return (torch.tensor(inputs,  dtype=torch.long, device=device), torch.tensor(targets, dtype=torch.long, device=device) )
    
    
def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(checkpoint, out)
    
    
def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint["iteration"]



import os 
import argparse

def arg_parse(): 
    p= argparse.ArgumentParser()
    
    p.add_argument("--train_path", type=str, default="/ext3/nyu-llm-reasoners-a1/data/results/train_tokens.npy")
    p.add_argument("--val_path",   type=str, default="/ext3/nyu-llm-reasoners-a1/data/results/valid_tokens.npy")
    p.add_argument("--run_name",type= str, default= "run" )
    p.add_argument("--lr", type = float, default= 3e-4)
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    

    p.add_argument("--beta1", type = float, default= 0.9)
    p.add_argument("--beta2", type = float, default=0.999 )
    p.add_argument("--epsilon", type = float, default= 1e-8)
    
    p.add_argument("--weight_decay", type = float, default= 0.1)
    
    p.add_argument("--checkpoint_dir", type=str, default="/ext3/nyu-llm-reasoners-a1/data/checkpoints")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a checkpoint to resume training from")
    
    #stuff fofr model
    p.add_argument("--vocab_size", type = int, default= 10000)
    p.add_argument("--context_length", type = int, default= 256)
    p.add_argument("--d_model", type = int, default= 512)
    p.add_argument("--d_ff", type = int, default= 1344)
    p.add_argument("--theta", type = int, default= 10000)
    p.add_argument("--num_layers", type = int, default= 4)
    p.add_argument("--num_heads", type = int, default= 16)
    p.add_argument("--tokens", type = int, default= 327680000)
   
    #EXPERIMENTATION
    # ABLATION WHERE each turns off one architectural component
    p.add_argument("--no_rmsnorm", action="store_true")
    p.add_argument("--post_norm",  action="store_true")
    p.add_argument("--no_rope",    action="store_true")
    p.add_argument("--use_silu",   action="store_true")
    
    
    p.add_argument("--eval_steps",    type=int, default=20)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--total_steps",  type=int,   default=5_000)
    p.add_argument("--warmup_steps", type=int,   default=200)
    p.add_argument("--grad_clip_eps",    type=float, default=1e-6)
    
    p.add_argument("--grad_clip_max_l2_norm", type = float, default= 1.0)
    
    return p.parse_args()  
    
from student.Transformer import TransformerLM




@torch.no_grad()
def validation_loss(model, val_data, args):
    model.eval()
    losses = []
    for _ in range(args.eval_steps):
        inputs, targets = get_batch(val_data, args.batch_size, args.context_length, args.device)
        loss = cross_entropy(
            model(inputs).view(-1, args.vocab_size),
            targets.view(-1),
        )
        losses.append(loss.item())
        
    model.train()

    return float(np.mean(losses))

import wandb
def init_wandb(args, model):
    wandb.init(
        project='llm-reasoners-a1',
        entity="saravargasmar-new-york-university",
        name=args.run_name,
        config=vars(args),
    )

    #log model size
    wandb.config.update({
        "n_parameters": sum(p.numel() for p in model.parameters())
    })

    
    wandb.watch(model, log="gradients", log_freq=100)
    
def main(): 
    args= arg_parse()
    train_data = np.load(args.train_path, mmap_mode="r")
    val_data   = np.load(args.val_path,   mmap_mode="r")

    
    model = TransformerLM(
        vocab_size=args.vocab_size, 
        context_length = args.context_length,
        d_model        = args.d_model,
        d_ff           = args.d_ff,
        num_layers     = args.num_layers,
        num_heads      = args.num_heads,
        theta     = args.theta,
         use_rmsnorm    = not args.no_rmsnorm,   
         pre_norm       = not args.post_norm,    
         use_rope       = not args.no_rope,      
         use_swiglu     = not args.use_silu,    
    ).to(args.device)
    
    n_params= sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}   using   device: {args.device}")
    print(f"Config: lr={args.lr}, batch={args.batch_size}, steps={args.total_steps}")
    init_wandb(args, model)
    optimiser= AdamW(
        model.parameters(), 
        lr= args.lr, 
        betas= (args.beta1, args.beta2), 
        weight_decay=args.weight_decay
    )
    
    start_step = 0
    #if there is a checkpoint to load 
    if args.checkpoint: 
        start_step = load_checkpoint(args.checkpoint_dir + args.checkpoint, model, optimiser)
        print("resumed at step {start_step}")
        
    model.train()
    
    start_time = time.time()
    
    for  step in range(start_step, args.total_steps):
        
        #update the lr accoring to scheduler cos
        lr_t = learning_rate_scheduler(
            t=step,
            lr_max=args.lr,
            lr_min=args.lr_min,
            Tw=args.warmup_steps,
            Tc=args.total_steps,
        )
        
        for group in optimiser.param_groups:
            group["lr"]= lr_t
            
            
        inputs, targets = get_batch(x= train_data,batch_size=args.batch_size,  context_length=args.context_length, device=args.device)
        
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, args.vocab_size), targets.view(-1))
        
        #backward
        optimiser.zero_grad()
        loss.backward()
        
        
        #gradient clipping: 
        gradient_clipping(parameters=model.parameters(), M =args.grad_clip_max_l2_norm )
        
        optimiser.step()
        
        wandb.log({"train/loss": loss.item(), "train/lr": lr_t}, step=step)

        if step %300 == 0 or step == args.total_steps - 1:
            val = validation_loss(model, val_data, args)
            time_elapsed = time.time() - start_time
            wandb.log({"val/loss": val}, step=step)
            print(
                f"step {step:6d} | "
                f"train {loss.item():.4f} | "
                f"val {val:.4f} | "
                f"lr {lr_t:.2e} | "
                f"{time_elapsed:.0f}s"
            )
            ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step:06d}.pt")
            save_checkpoint(model, optimiser, step, ckpt_path)

            
    
    

    
    
    
    
    
    
    
    
    
    #training loop 
    
    #configure and control various modele and optimizer hyperparams
    #memoryefficeitn loading of training and validation datasets with mp_memmap 
    
    #serializing checkpoints to userprovided path 


if __name__ == "__main__":
    
    main() #training loop

    
# if __name__== "__main__": 
#     weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
#     opt = SGD([weights], lr=1)
    
#     for t in range(100):
#         opt.zero_grad() # Reset the gradients for all learnable parameters. loss = (weights**2).mean() # Compute a scalar loss value. print(loss.cpu().item())
#         loss = (weights**2).mean()
#         print(loss.cpu().item())
#         loss.backward() # Run backward pass, which computes gradients. opt.step() # Run optimizer step.
#         opt.step()
        
        
        
        
        
