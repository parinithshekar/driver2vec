import torch
import pandas as pd
import numpy as np

from dataset import helper
from models.d2v import D2V

def main():
    input_channels = 31
    input_length = 1000
    output_channels = 32
    kernel_size = 16
    dilation_base = 2

    inputs = helper.sample_input()
    inputs = torch.Tensor(inputs.to_numpy())
    inputs = inputs.reshape(1, input_channels, input_length)
    splits = torch.split(inputs, input_channels, dim=1)
      
    model = D2V(input_channels, input_length, output_channels, kernel_size, dilation_base)
    model(inputs)
    return
    inputs = splits[0]
    wvlt_inputs = splits[1]
    wvlt_inputs = inputs
    wvlt_inputs_1 = torch.split(wvlt_inputs,
                                input_length // 2,
                                dim=1)[0]
    wvlt_inputs_2 = torch.split(wvlt_inputs,
                                input_length // 2,
                                dim=1)[1]
    bsize = inputs.size()[0]
    wvlt_out1 = wvlt_inputs_1.reshape(bsize, -1, 1).squeeze()
    wvlt_out2 = wvlt_inputs_2.reshape(bsize, -1, 1).squeeze()
    

if __name__ == "__main__":
    main()