import torch
from einops import rearrange, einsum
import torch.nn as nn


def softmax(x , ith_dim) -> torch.Tensor:
    # trick of subtracting max val in ith dim from all eelemtns for numerical stability
    x = x - x.max(dim=ith_dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=ith_dim, keepdim=True)
    

def scaled_dot_product_attention(Q, K, V, mask=None) -> torch.Tensor: 
    dk= Q.shape[-1]
    
    #scores (..., seq len Q , seqlen K)
    scores = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / (dk ** 0.5)
    
    if mask is not None: 
        scores= scores.masked_fill(mask ==False, float('-inf'))
        
    #apply softmax
    attn_weights = softmax(scores, ith_dim=-1)

    #weighted sum attention : for every query a weighted combination of value vectors 
    attention = einsum(attn_weights, V, "... q k, ... k dv -> ... q dv")
    return attention
    



# Einsum notation:
# its basically a way of 
class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        '''
        Parameters: 
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        
        Define Parameters 
        '''
        #run the superclass constructor 
        ## nn module basically tracks parameters and calling this superclass constructor 
        super().__init__()
         
        std = (2.0 / (in_features + out_features)) ** 0.5
        self.W= nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.W, mean=0.0,std=std,a=-3 * std,b=3 * std)    
            
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        #Apply the linear transformation to the  input
        output= torch.einsum("...i,oi->...o", x, self.W) #... is for the leaing dimensions that could be batch and sequence
        
        return output
        
    
#Remembre the embedding matrix is like a look up table for every vocab so it has size vocab by dmodel 
#since we want a vector per vocab 
class Embedding(nn.Module): 
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        #num_embeddings is vocab size
        #dim is vocab x dmodel because each row is some token embedding vector 
        self.embedding_matrix= nn.Parameter(torch.empty(num_embeddings,embedding_dim, device=device, dtype=dtype ))
        torch.nn.init.trunc_normal_(self.embedding_matrix, mean=0.0,std=(1)**0.5,a=-3 ,b=3 )    
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        #gives you the row vectors of the tokens you asked for 
        #which are the embeddings per token
        return self.embedding_matrix[token_ids]
        
#PRENORM: LAYER NORMALIZATION BEFORE EACH SUBLAYER    (BEFORE ATTENTION AND FFN)
#INTUITION: There is a clean “residual stream” without any normalization going
#   from the input embeddings to the final output of the Transformer, which is purported to improve gradient flow. 
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        '''
        :param d_model: hidden dim
        :param eps: epsilon for numerical stability
        '''
        super().__init__()
        self.eps=eps
        #gain parameter thats learnable
        self.g = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        #Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms= torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        rms_norm= (x/rms)* self.g
        
        return rms_norm.to(in_dtype)
        
class positionwise_ffn(nn.Module):
    def __init__(self,dff,  dmodel):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(dff, dmodel))
        self.w2 = nn.Parameter(torch.empty(dmodel, dff))
        self.w3 = nn.Parameter(torch.empty(dff, dmodel))
        
        self.dff= (8/3)*dmodel 
        
        
    def forward(self, x): 
        x1 = torch.einsum("...m,fm->...f", x, self.w1)  
        x3 = torch.einsum("...m,fm->...f", x, self.w3)
        silu = x1 * torch.sigmoid(x1)        
        out = torch.einsum("...f,mf->...m", silu * x3, self.w2)
        return out

    
    
#Relative Positional Embeddings
class RotaryPositionalEmbedding(nn.Module): 
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        
        #indices of each dimension pair
        k = torch.arange(0, d_k // 2, device=device)
        self.theta= theta 
        inv_freq = 1.0 / (theta ** (2 * k / d_k))
        
        positions = torch.arange(max_seq_len, device=device).float()  
        angles = einsum(positions, inv_freq, "i, j -> i j")  # 2) compute outer product
        
        #regirster buffers
        self.register_buffer("cos_vals", torch.cos(angles), persistent=False)
        self.register_buffer("sin_vals", torch.sin(angles), persistent=False)

        
        
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        
        #look up precomputed cos and sin
        cos = self.cos_vals[token_positions] 
        sin = self.sin_vals[token_positions] 
        
        # split into even and odd
        x1 = x[..., 0::2]  
        x2 = x[..., 1::2] 
        
        # rotation to each pair x1 and x2
        x_rotated_1 = x1 * cos - x2 * sin
        x_rotated_2 = x1 * sin + x2 * cos
        
        # join back
        out = torch.stack([x_rotated_1, x_rotated_2], dim=-1)  
        return out.flatten(-2) 
        
    

            
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float = None, max_seq_len: int = None, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dk = dv = d_model / h
        
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None
        # parameter matrices 
        self.W_Q = LinearLayer(d_model, d_model, device=device, dtype=dtype)
        self.W_K = LinearLayer(d_model, d_model, device=device, dtype=dtype)
        self.W_V = LinearLayer(d_model, d_model, device=device, dtype=dtype)
        self.W_O = LinearLayer(d_model, d_model, device=device, dtype=dtype)   


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
    
        #get q, k, v projections
        Q = self.W_Q(x)  
        K = self.W_K(x)  
        V = self.W_V(x) 
        
        # split into heads
        Q = Q.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch, heads, seq_len, d_k)
        K = K.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch, heads, seq_len, d_k)
        V = V.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch, heads, seq_len, d_k)
        
        if self.rope is not None:
            token_positions_exp = token_positions.unsqueeze(0).unsqueeze(0).expand(batch, self.num_heads, seq_len)
            Q = self.rope(Q, token_positions_exp)
            K = self.rope(K, token_positions_exp)
                
        mask = ~torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)

        attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
    
        return self.W_O(attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float = None, max_seq_len: int = None, device=None, dtype=None):
        super().__init__()
        
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)
        
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = positionwise_ffn(dff=d_ff, dmodel=d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
    
    
    
    
#PUTTING IT ALL TOGETHER
class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, theta: float = None, device=None, dtype=None):
        super().__init__()
        
        #TOKEN IDS TO EMBEDDING VECTORS
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # STACK BLOCKS
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, 
                           theta=theta, max_seq_len=context_length, device=device, dtype=dtype)
            for i in range(num_layers)
        ])
        
        #normalization layer before last projection
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        
        # project from d_model to vocab_size to get next token logits
        self.lm_head = LinearLayer(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
    
        hidden_states = self.token_embeddings(token_ids)  # (batch_size, sequence_length, d_model)
    
        sequence_length = token_ids.shape[1]
        token_positions = torch.arange(sequence_length, device=token_ids.device)
        
        # pass through each transformer block
        for transformer_block in self.layers:
            hidden_states = transformer_block(hidden_states, token_positions)
        
        # normalize and project to vocabulary
        hidden_states = self.ln_final(hidden_states)
        next_token_logits = self.lm_head(hidden_states)  # (batch_size, sequence_length, vocab_size)
        
        return next_token_logits