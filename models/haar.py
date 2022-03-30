from audioop import bias
import torch
from torch import nn
import pywt

class HaarWavelet(nn.Module):
    def __init__(self, input_channels, input_length, output_length):
        super(HaarWavelet, self).__init__()
        self.input_channels = input_channels
        self.input_length = input_length
        self.wavelet_size = input_channels * input_length // 2
        self.fc1 = nn.Linear(self.wavelet_size, output_length, bias=True)
        self.fc2 = nn.Linear(self.wavelet_size, output_length, bias=True)
        nn.init.xavier_normal(self.fc1.weight)
        nn.init.xavier_normal(self.fc2.weight)
    
    def forward(self, input):
        # get wavelet transform
        # haar1, haar2 = torch.split(input, self.input_length//2, dim=1)
        # wa, wd = pywt.dwt(input.cpu().detach().numpy(), wavelet="haar")
        haar1, haar2 = pywt.dwt(input.cpu().detach().numpy(), wavelet="haar")
        haar1, haar2 = torch.flatten(torch.Tensor(haar1)), torch.flatten(torch.Tensor(haar2))
        out1, out2 = self.fc1(haar1), self.fc2(haar2)
        return torch.cat([out1, out2])
