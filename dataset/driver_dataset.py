import random as rd
import torch
from torch.utils.data import Dataset
import numpy as np

from .helper import extract_dataset

class DriverDataset(Dataset):
    def __init__(self, number_of_users, section_size, modality, train_ratio, drop_feature_groups=[]):
        self.raw_dataset = extract_dataset(
            modality,
            train_ratio=train_ratio,
            section_size=section_size,
            drop_feature_groups=drop_feature_groups
        )
        self.users = set([i+1 for i in range(number_of_users)])

        self.dataset = []
        
        # Prepare list of training entries
        for driver in list(self.raw_dataset.keys()):
            l = len(self.raw_dataset[driver])
            iset = set(range(l))
            for i in range(l):
                anch = self.raw_dataset[driver][i]
                pi = rd.choice(list(iset - {i}))
                pos = self.raw_dataset[driver][pi]
                neg = rd.choice(self.raw_dataset[rd.choice(list(self.users - {driver}))])
                self.dataset.append({ "driver": driver, "triplet": (anch, pos, neg) })
        print(len(self.dataset))

        
    def __len__(self):
        # Return size of the dataset
        return len(self.dataset)

    def __getitem__(self, index):
        entry = self.dataset[index]
        return entry['triplet']
    
    # def __getitem__(self, index):
    #     anchor_user = rd.choice(list(self.users))
    #     negative_user = rd.choice(list(self.users - {anchor_user}))
        
    #     # Get triplets
    #     anchor = rd.choice(self.raw_dataset[anchor_user])
    #     positive = rd.choice(self.raw_dataset[anchor_user])
    #     while torch.equal(anchor, positive):
    #         positive = rd.choice(self.raw_dataset[anchor_user])
    #     negative = rd.choice(self.raw_dataset[negative_user])

    #     return anchor, positive, negative
    
    def sample(self):
        return list(self.raw_dataset.values())[0][0]
    
    def get_lightgbm_inputs(self, drivers={1, 2}, binary=True):
        inputs = []
        labels = []
        for driver in drivers:
            inputs += self.raw_dataset[driver]
            labels += [driver for _ in range(len(self.raw_dataset[driver]))]
        
        if (len(drivers)==2 and binary):
            labels = [0 if driver==min(labels) else 1 for driver in labels]
        
        inputs = torch.stack(inputs)
        labels = np.array(labels)
        return inputs, labels
