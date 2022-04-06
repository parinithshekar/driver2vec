from cgi import test
from lightgbm import train
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
import numpy as np
import random as rd
import os
from datetime import datetime

from tqdm import tqdm
from tabulate import tabulate

from dataset.driver_dataset import DriverDataset
from dataset import features, helper
from models.d2v import D2V
from models.lightGBM import LightGBM

from train import build_train_save_tcn
from classify import train_classify_score_lgbm

torch.manual_seed(9090)
np.random.seed(7080)
rd.seed(4496)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CURRENT_TIME = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(DIR, "trained_models")

def test():
    # x = pd.read_csv("./dataset/sample_data/user_0001/user_0001_highway.csv")
    # x = x.drop(columns=["Unnamed: 0"])
    # x = x.drop(columns=['FOG', 'FOG_LIGHTS', 'FRONT_WIPERS','HEAD_LIGHTS','RAIN', 'REAR_WIPERS', 'SNOW'])
    # print(x.columns)

    # x = helper.extract_dataset(modality="test", drop_feature_groups=[features.ACCELERATION])
    # print(x[1][0].shape)

    train_dataset = DriverDataset(number_of_users=5, section_size=200, modality='train', train_ratio=0.8, drop_feature_groups=[features.groups['DISTANCE_INFORMATION']])
    input_channels = train_dataset.sample().shape[0]
    print(input_channels)

def main():

    all_results = {}

    for feature_group, features_list in features.groups.items():
        build_train_save_tcn(drop_feature_groups=[features_list])
        accuracy_score = train_classify_score_lgbm(drop_feature_groups=[features_list])
        all_results[feature_group] = accuracy_score
    
    build_train_save_tcn()
    accuracy_score = train_classify_score_lgbm()
    all_results['ALL_FEATURES'] = accuracy_score

    table = tabulate([[k, v] for k, v in all_results.items()], headers=["DROPPED FEATURE GROUP", "PAIRWISE ACCURACY"])
    print(table)

if __name__ == "__main__":
    main()