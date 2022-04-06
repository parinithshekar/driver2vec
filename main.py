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

from dataset.driver_dataset import DriverDataset
from dataset import features, helper
from models.d2v import D2V
from models.lightGBM import LightGBM

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

    x = helper.extract_dataset(modality="test", drop_feature_groups=[features.ACCELERATION])
    print(x[1][0].shape)

def main():
    test()

if __name__ == "__main__":
    main()