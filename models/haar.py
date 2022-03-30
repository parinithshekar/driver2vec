from audioop import bias
import torch
from torch import nn

class HaarWavelet(nn.Module):
    def __init__(self, input_channels, input_length, output_length):
        super(HaarWavelet, self).__init__()
        self.input_channels = input_channels
        self.input_length = input_length
        self.input_size = input_channels * input_length // 2
        self.fc1 = nn.Linear(self.input_size, output_length, bias=True)
        self.fc2 = nn.Linear(self.input_size, output_length, bias=True)
    
    def forward(self, input):
        output = []
        for n_input in input:
            # get wavelet transform
            print(n_input.shape)
            haar1, haar2 = torch.split(n_input, self.input_length//2, dim=1)
            haar1, haar2 = torch.flatten(haar1), torch.flatten(haar2)
            out1, out2 = self.fc1(haar1), self.fc2(haar2)
            
            print(out1.shape, out2.shape)