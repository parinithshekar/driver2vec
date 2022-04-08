import os
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random as rd

from dataset.driver_dataset import DriverDataset
from models.d2v import D2V

torch.manual_seed(9090)
np.random.seed(7080)
rd.seed(4496)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CURRENT_TIME = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(DIR, "trained_models")
# LATEST_MODEL = os.path.join(SAVE_DIR, f"latest.pt")

def build_train_save_tcn(drop_feature_groups=[], save_prefix=""):
    save_latest = True

    if (save_prefix == ""):
        latest_model = os.path.join(SAVE_DIR, f"latest.pt")
    else:
        latest_model = os.path.join(SAVE_DIR, f"{save_prefix}_latest.pt")
    
    input_length = 200
    output_channels = 32
    kernel_size = 16
    dilation_base = 2
    batch_size = 16

    epochs = 70
    train_ratio = 0.8

    train_dataset = DriverDataset(number_of_users=5, section_size=input_length, modality='train', train_ratio=train_ratio, drop_feature_groups=drop_feature_groups)
    # test_dataset = DriverDataset(number_of_users=5, section_size=input_length, modality='test', train_ratio=train_ratio)
    
    # input_channels = 31
    input_channels = train_dataset.sample().shape[0]
    
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = D2V(input_channels, input_length, output_channels, kernel_size, dilation_base)
    model = model.to(device)

    # Hyperparameters from the paper
    optimizer = optim.Adam(model.parameters(), lr=0.0004, weight_decay=0.975)
    triplet_loss = nn.TripletMarginLoss(margin=1)

    model.train()
    for epoch in range(epochs):
        running_loss = []
        for step, (anchor, positive, negative) in enumerate(loader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)

            loss = triplet_loss(anchor_output, positive_output, negative_output)
            loss.backward()
            optimizer.step()
    
            running_loss.append(loss.cpu().detach().numpy())

        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))

    # save_path = os.path.join(SAVE_DIR, f"d{CURRENT_TIME}_e{epochs}_b{batch_size}_l{input_length}.pt")
    # torch.save(model.state_dict(), save_path)

    if(save_latest):
        torch.save(model.state_dict(), latest_model)
