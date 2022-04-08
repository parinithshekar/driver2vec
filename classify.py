
import torch
import itertools as itt
import numpy as np
import random as rd
import os
from datetime import datetime

from dataset.driver_dataset import DriverDataset
from models.d2v import D2V
from models.lightGBM import LightGBM

torch.manual_seed(9090)
np.random.seed(7080)
rd.seed(4496)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(DIR, "trained_models")
# LATEST_MODEL = os.path.join(SAVE_DIR, f"latest.pt")

def train_classify_score_lgbm(drop_feature_groups=[], model_prefix=""):

    if (model_prefix == ""):
        latest_model = os.path.join(SAVE_DIR, f"latest.pt")
    else:
        latest_model = os.path.join(SAVE_DIR, f"{model_prefix}_latest.pt")

    input_length = 200
    output_channels = 32
    kernel_size = 16
    dilation_base = 2

    train_ratio = 0.8

    train_dataset = DriverDataset(number_of_users=5, section_size=input_length, modality='train', train_ratio=train_ratio, drop_feature_groups=drop_feature_groups)
    test_dataset = DriverDataset(number_of_users=5, section_size=input_length, modality='test', train_ratio=train_ratio, drop_feature_groups=drop_feature_groups)
    
    # input_channels = 31
    input_channels = train_dataset.sample().shape[0]

    model = D2V(input_channels, input_length, output_channels, kernel_size, dilation_base)
    model.load_state_dict(torch.load(latest_model))
    model = model.to(device)

    model.eval()

    users = [1, 2, 3, 4, 5]
    accuracies = []
    for driver_pair in itt.combinations(users, 2):
        # Get embeddings for lightGBM
        lgbm_inputs, lgbm_labels = train_dataset.get_lightgbm_inputs(drivers=driver_pair)
        lgbm_inputs = lgbm_inputs.to(device)
        embeddings = model(lgbm_inputs)
        embeddings = embeddings.cpu().data.numpy()
        # Train lightGBM
        classifier = LightGBM()
        classifier.train(embeddings,lgbm_labels)

        # Predict outputs
        lgbm_test_inputs, lgbm_test_labels = test_dataset.get_lightgbm_inputs(drivers=driver_pair)
        lgbm_test_inputs = lgbm_test_inputs.to(device)
        lgbm_test_embeddings = model(lgbm_test_inputs)
        lgbm_test_embeddings = lgbm_test_embeddings.cpu().data.numpy()
        predictions = classifier.predict(lgbm_test_embeddings)
        
        binary_predictions = [0 if y<0.5 else 1 for y in predictions]

        errors = np.abs(np.array(binary_predictions)-np.array(lgbm_test_labels))
        accuracy = 1 - np.sum(errors)/errors.size
        accuracies.append(accuracy)
    
    mean_accuracy = np.mean(accuracies)
    return mean_accuracy
