from torch import nn
import torch

from models.tcn import TCN
from models.haar import HaarWavelet

class D2V(nn.Module):
    def __init__(self, input_channels, input_length):
        super(D2V, self).__init__()
        self.tcn = TCN()
        self.wavelet = HaarWavelet(input_channels, input_length, output_length=15)
    
    def forward(self, input):
        self.wavelet(input)
