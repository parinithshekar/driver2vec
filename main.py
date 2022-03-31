import torch
import pandas as pd
import numpy as np

from dataset import helper
from models.d2v import D2V

def main():
    filenames = ["user_0001/user_0001_highway.csv", "user_0001/user_0001_suburban.csv", "user_0001/user_0001_tutorial.csv", "user_0001/user_0001_urban.csv", "user_0002/user_0002_highway.csv", "user_0002/user_0002_suburban.csv", "user_0002/user_0002_tutorial.csv", "user_0002/user_0002_urban.csv", "user_0003/user_0003_highway.csv", "user_0003/user_0003_suburban.csv", "user_0003/user_0003_tutorial.csv", "user_0003/user_0003_urban.csv", "user_0004/user_0004_highway.csv", "user_0004/user_0004_suburban.csv", "user_0004/user_0004_tutorial.csv", "user_0004/user_0004_urban.csv", "user_0005/user_0005_highway.csv", "user_0005/user_0005_suburban.csv", "user_0005/user_0005_tutorial.csv", "user_0005/user_0005_urban.csv"]
    X = {0: {0: [], 1: [], 2: [], 3: []}, 1: {0: [], 1: [], 2: [], 3: []}, 2: {0: [], 1: [], 2: [], 3: []}, 3: {0: [], 1: [], 2: [], 3: []}, 4: {0: [], 1: [], 2: [], 3: []} }
    y = {0: {0: [], 1: [], 2: [], 3: []}, 1: {0: [], 1: [], 2: [], 3: []}, 2: {0: [], 1: [], 2: [], 3: []}, 3: {0: [], 1: [], 2: [], 3: []}, 4: {0: [], 1: [], 2: [], 3: []} }
    X_dict = helper.extract_data(filenames, X)
    
    input_channels = 31
    input_length = 1000
    output_channels = 32
    kernel_size = 16
    dilation_base = 2
    filename = "user_0001/user_0001_highway.csv"

    inputs = helper.sample_input(filename)
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