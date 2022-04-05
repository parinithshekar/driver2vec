import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
import numpy as np
import random as rd
import os

from tqdm import tqdm

from dataset.driver_dataset import DriverDataset
from dataset import helper
from models.d2v import D2V

torch.manual_seed(9090)
np.random.seed(7080)
rd.seed(4496)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(DIR, "trained_models")

def main():
    # filenames = ["user_0001/user_0001_highway.csv", "user_0001/user_0001_suburban.csv", "user_0001/user_0001_tutorial.csv", "user_0001/user_0001_urban.csv", "user_0002/user_0002_highway.csv", "user_0002/user_0002_suburban.csv", "user_0002/user_0002_tutorial.csv", "user_0002/user_0002_urban.csv", "user_0003/user_0003_highway.csv", "user_0003/user_0003_suburban.csv", "user_0003/user_0003_tutorial.csv", "user_0003/user_0003_urban.csv", "user_0004/user_0004_highway.csv", "user_0004/user_0004_suburban.csv", "user_0004/user_0004_tutorial.csv", "user_0004/user_0004_urban.csv", "user_0005/user_0005_highway.csv", "user_0005/user_0005_suburban.csv", "user_0005/user_0005_tutorial.csv", "user_0005/user_0005_urban.csv"]
    # X = {0: {0: [], 1: [], 2: [], 3: []}, 1: {0: [], 1: [], 2: [], 3: []}, 2: {0: [], 1: [], 2: [], 3: []}, 3: {0: [], 1: [], 2: [], 3: []}, 4: {0: [], 1: [], 2: [], 3: []} }
    # y = {0: {0: [], 1: [], 2: [], 3: []}, 1: {0: [], 1: [], 2: [], 3: []}, 2: {0: [], 1: [], 2: [], 3: []}, 3: {0: [], 1: [], 2: [], 3: []}, 4: {0: [], 1: [], 2: [], 3: []} }
    # X_dict = helper.extract_data(filenames, X)

    input_channels = 31
    input_length = 200
    output_channels = 32
    kernel_size = 16
    dilation_base = 2
    batch_size = 16

    epochs = 200
    train_ratio = 0.8

    train_dataset = DriverDataset(number_of_users=5, section_size=input_length, modality='train', train_ratio=train_ratio)
    test_dataset = DriverDataset(number_of_users=5, section_size=input_length, modality='test', train_ratio=train_ratio)
    
    train_labels = train_dataset.generate_labels()
    test_labels = test_dataset.generate_labels()
    
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = D2V(input_channels, input_length, output_channels, kernel_size, dilation_base)

    # Hyperparameters from the paper
    optimizer = optim.Adam(model.parameters(), lr=0.0004, weight_decay=0.975)
    triplet_loss = nn.TripletMarginLoss(margin=1)

    model.train()
    for epoch in tqdm(range(epochs), desc='Epochs'):
        running_loss = []
        for step, (anchor, positive, negative) in enumerate(tqdm(loader, desc="Training", leave=False)):
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

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, os.path.join(SAVE_DIR, f"e{epochs}_b{batch_size}_l{input_length}.pth"))
    return
    
    
    filename1 = "user_0001/user_0001_highway.csv"
    filename2 = "user_0001/user_0001_suburban.csv"

    input1 = torch.Tensor((helper.sample_input(filename1)).to_numpy())
    input2 = torch.Tensor((helper.sample_input(filename2)).to_numpy())

    input1 = input1.reshape(1, input_channels, input_length)
    input2 = input2.reshape(1, input_channels, input_length)
    inputs = torch.cat([input1, input2])

    model = D2V(input_channels, input_length, output_channels, kernel_size, dilation_base)
    out = model(inputs)

    print(out)
    
    return
    

if __name__ == "__main__":
    main()