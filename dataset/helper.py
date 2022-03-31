import pandas as pd
import numpy as np
import torch

import os
    
DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, "sample_data")

def sample_input(filename):
    x = pd.read_csv(os.path.join(DATA, filename))
    x = x.drop(columns=["Unnamed: 0"])
    x = x.drop(columns=['FOG', 'FOG_LIGHTS', 'FRONT_WIPERS','HEAD_LIGHTS','RAIN', 'REAR_WIPERS', 'SNOW'])
    return x

def extract_data(filenames, X_dict):
    for i in range(5):
        for j in range(4):
            inputs = sample_input(filenames[4*i+j])
            inputs = torch.Tensor(inputs.to_numpy())
            
            split = torch.split(inputs,200) # split into 200 sample arrays
            stack = torch.stack(list(split)[:int(inputs.shape[0]/200)],dim=0) # stack the tuples into a tensor
            X_dict[i][j] = stack
    return X_dict

def split_data(X_dict):
    X_test = []
    X_train = []
    y_test = []
    y_train = []
    for i in range(5):
        for j in range(4):
            seed = 0 # we can define this randomly
            if j != 2:
                X_test.append(X_dict[i][j][seed,:,:].reshape(1,X_dict[i][j].shape[1],X_dict[i][j].shape[2]))
                X_train.append(X_dict[i][j][seed+1:,:,:])
            else:
                X_train.append(X_dict[i][j])
                
    return X_test, X_train, y_test, y_train
