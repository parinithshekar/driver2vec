from genericpath import isfile
import pandas as pd
import numpy as np
import torch

import os
    
DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, "sample_data")

def read_data_from_filepath(filepath):
    x = pd.read_csv(os.path.join(DATA, filepath))
    x = x.drop(columns=["Unnamed: 0"])
    x = x.drop(columns=['FOG', 'FOG_LIGHTS', 'FRONT_WIPERS','HEAD_LIGHTS','RAIN', 'REAR_WIPERS', 'SNOW'])
    return x

def read_data_from_file(filename):
    x = pd.read_csv(os.path.join(DATA, filename))
    x = x.drop(columns=["Unnamed: 0"])
    x = x.drop(columns=['FOG', 'FOG_LIGHTS', 'FRONT_WIPERS','HEAD_LIGHTS','RAIN', 'REAR_WIPERS', 'SNOW'])
    return x

def extract_data(filenames, X_dict):
    for i in range(5):
        for j in range(4):
            inputs = read_data_from_file(filenames[4*i+j])
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

def section_data(df, section_size=200):
    all_data = torch.Tensor(df.to_numpy())
    sections = []
    for i in range(int((all_data.shape[0]-200)/40)+1):
        sections.append(torch.transpose(all_data[40*i:200+40*i,:], 0, 1))
    #sections = [torch.transpose(s, 0, 1) for s in list(torch.split(all_data, section_size)) if s.shape[0]==section_size]
    return sections

# TODO: support train, cross-validation and test splits
def extract_dataset(modality, train_ratio=0.8, section_size=200):
    train_dataset = {}
    test_dataset = {}
    for user_i in range(5):
        user = user_i + 1
        user_dir = f"user_{user:04}"
        user_data_path = os.path.join(DATA, user_dir)
        user_files = [f for f in os.listdir(user_data_path) if isfile(os.path.join(user_data_path, f))]
        for f in user_files:
            filepath = os.path.join(user_data_path, f)
            df = read_data_from_file(filepath)
            sections = section_data(df, section_size=section_size)
            train_size = round(train_ratio*len(sections))
            if user in train_dataset:
                train_dataset[user] += sections[:train_size]
                test_dataset[user] += sections[train_size:]
            else:
                train_dataset[user] = sections[:train_size]
                test_dataset[user] = sections[train_size:]

    # print(dataset.keys())
    # for k in dataset:
    #     print(len(dataset[k]))
    
    if modality == 'train':
        return train_dataset
    else:
        return test_dataset
        
