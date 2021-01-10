import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline
# 加载数据并设置标签
data_complaint = pd.read_csv('data/complaint1700.csv')
data_complaint['label'] = 0
data_non_complaint = pd.read_csv('data/noncomplaint1700.csv')
data_non_complaint['label'] = 1

# 将抱怨和不抱怨的两个数据合成一块
data = pd.concat([data_complaint, data_non_complaint], axis=0).reset_index(drop=True)

# 删除 'airline' 列
data.drop(['airline'], inplace=True, axis=1)

# 展示随机的5个样本
data.sample(5)

from sklearn.model_selection import train_test_split

X = data['tweet'].values
y = data['label'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=2020)

# Load test data
test_data = pd.read_csv('data/test_data.csv')

# Keep important columns
test_data = test_data[['id', 'tweet']]

# Display 5 samples from the test data
test_data.sample(5)


def get_data():
    return data


def get_test_data():
    return test_data

# 使用GPU、CPU来训
