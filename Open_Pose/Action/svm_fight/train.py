import numpy as np
import pandas as pd

def load_data(features):
    raw_neg=pd.read_csv('data.csv',header=0)
    dataset=raw_neg.values
    X=dataset[0:1000,0:36]
    Y=dataset[0:1000,36]
    print(X)
    print(Y)
    print(X.shape)
    print(Y.shape)


def train(dir_name):
    labels=[]
    features=[]
