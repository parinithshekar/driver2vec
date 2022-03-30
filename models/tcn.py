from torch import nn
import torch

class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()
        self.layers = []
        
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x