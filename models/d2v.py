from torch import nn
import torch

from models.tcn import TCN
from models.haar import HaarWavelet

class D2V(nn.Module):
    def __init__(self, input_channels, input_length, output_channels, kernel_size=16, dilation_base = 2):
        super(D2V, self).__init__()
        self.tcn = TCN(input_channels,output_channels ,kernel_size, dilation_base=dilation_base, dropout = 0.2)
        self.wavelet = HaarWavelet(input_channels, input_length, output_length=15)
    
    def forward(self, input):
        # print(f"Input: {input.shape}")
        tcn_output = self.tcn(input)[:, :, -1]
        # print(f"TCN output: {tcn_output.shape}")
        wavelet_output = self.wavelet(input)
        # print(f"Wavelet output: {wavelet_output.shape}")
        return torch.cat((tcn_output, wavelet_output), dim=1)
