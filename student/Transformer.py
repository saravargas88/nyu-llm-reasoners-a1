import torch
from einops import rearrange, einsum
import torch.nn as nn

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
        self.W1= nn.Parameter(torch.empty(dff, dmodel))
        self.W2 = nn.Parameter(torch.empty(dmodel, dff))
        self.W3= nn.Parameter(torch.empty(dff, dmodel))
        
        self.dff= (8/3)*dmodel 
        
        
    def forward(self, x): 
        x1= einsum("...m,fm->...f" , x , self.W1)  
        x3= einsum("...m,fm->...f",x, self.W3)
       
        silu= x1 * torch.sigmoid(x1)        
        in_mult= silu * x3
        
        out= einsum("...f,mf->...m", in_mult, self.W2)
        return out
    
    
#Relative Positional Embeddings
class RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=None): 
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        
    
    
    
        
            
            
            