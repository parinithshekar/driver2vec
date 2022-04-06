from genericpath import isfile
import pandas as pd
import numpy as np
import torch

import os

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, "sample_data")

def read_data_from_filepath(filepath, drop_feature_groups=[]):
    x = pd.read_csv(filepath)
    x = x.drop(columns=["Unnamed: 0"])
    x = x.drop(columns=['FOG', 'FOG_LIGHTS', 'FRONT_WIPERS','HEAD_LIGHTS','RAIN', 'REAR_WIPERS', 'SNOW'])
    for feature_group in drop_feature_groups:
        x = x.drop(columns=feature_group)
    return x

def section_data(df, section_size=200):
    all_data = torch.Tensor(df.to_numpy())
    sections = []
    for i in range(int((all_data.shape[0]-200)/40)+1):
        sections.append(torch.transpose(all_data[40*i:200+40*i,:], 0, 1))
    #sections = [torch.transpose(s, 0, 1) for s in list(torch.split(all_data, section_size)) if s.shape[0]==section_size]
    return sections

def extract_dataset(modality, train_ratio=0.8, section_size=200, drop_feature_groups=[]):
    train_dataset = {}
    test_dataset = {}
    for user_i in range(5):
        user = user_i + 1
        user_dir = f"user_{user:04}"
        user_data_path = os.path.join(DATA, user_dir)
        user_files = [f for f in os.listdir(user_data_path) if isfile(os.path.join(user_data_path, f))]
        for f in user_files:
            filepath = os.path.join(user_data_path, f)
            df = read_data_from_filepath(filepath, drop_feature_groups=drop_feature_groups)
            sections = section_data(df, section_size=section_size)
            train_size = round(train_ratio*len(sections))
            if user in train_dataset:
                train_dataset[user] += sections[:train_size]
                test_dataset[user] += sections[train_size:]
            else:
                train_dataset[user] = sections[:train_size]
                test_dataset[user] = sections[train_size:]
    
    if modality == 'train':
        return train_dataset
    else:
        return test_dataset
        
