import torch
from einops import rearrange, einsum
import torch.nn as nn





# Einsum notation:
# its basically a way of 
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        '''
        Parameters: 
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        
        
        '''
        #run the superclass constructor 
        ## nn module basically tracks parameters and calling this superclass constructor 
        
        super().__init__() 
        #own 
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        #Apply the linear transformation to the  input
        pass