# -*- coding: utf-8 -*-
"""
@Auth : ZQB
@Time : 2024/9/28 3:48 PM
@File :bank.py
"""
import os

import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
from .base_dataset import BaseDataset

class Bank(BaseDataset):
    def __init__(self, cfg, rank, num_clients, num_models, dataset_mode):
        super(Bank, self).__init__(cfg, rank, num_clients, num_models, dataset_mode)
        self._csv_path = cfg.path

        self._get_train_aux_test_data_tensor()

    def _load_data_from_csv(self):
        data = pd.read_csv(self._csv_path)
        data = data.drop(['customer_id'], axis=1)
        data['country'] = LabelEncoder().fit_transform(data['country'])
        data['gender'] = LabelEncoder().fit_transform(data['gender'])

        y = data.iloc[:, -1]
        x = data.iloc[:, :-1]

        # One hot label
        enc = OneHotEncoder()
        y = enc.fit_transform(y.values.reshape(-1, 1)).toarray()
        y = y.astype(int)

        return x, y