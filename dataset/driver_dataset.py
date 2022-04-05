import random as rd
import torch
from torch.utils.data import Dataset, DataLoader

from .helper import extract_dataset

class DriverDataset(Dataset):
    def __init__(self, number_of_users, section_size, modality, train_ratio):
        self.dataset = extract_dataset(modality, train_ratio=train_ratio, section_size=section_size)
        self.users = set([i+1 for i in range(number_of_users)])

    def __len__(self):
        # Return size of the dataset
        return sum([len(s) for s in list(self.dataset.values())])

    def __getitem__(self, index):
        anchor_user = rd.choice(list(self.users))
        negative_user = rd.choice(list(self.users - {anchor_user}))
        
        # Get triplets
        anchor = rd.choice(self.dataset[anchor_user])
        positive = rd.choice(self.dataset[anchor_user])
        while torch.equal(anchor, positive):
            positive = rd.choice(self.dataset[anchor_user])
        negative = rd.choice(self.dataset[negative_user])

        # Change from (31, 200) to (1, 31, 200)
        # anchor = torch.unsqueeze(anchor, 0)
        # positive = torch.unsqueeze(positive, 0)
        # negative = torch.unsqueeze(negative, 0)

        return anchor, positive, negative
