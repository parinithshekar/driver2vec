import pandas as pd
import numpy as np

import os
    
DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, "sample_data")

def sample_input():
    x = pd.read_csv(os.path.join(DATA, "user_0001/user_0001_highway.csv"))
    x = x.drop(columns=["Unnamed: 0"])
    x = x.drop(columns=['FOG', 'FOG_LIGHTS', 'FRONT_WIPERS','HEAD_LIGHTS','RAIN', 'REAR_WIPERS', 'SNOW'])
    return x
