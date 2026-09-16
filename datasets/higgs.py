# -*- coding: utf-8 -*-
"""
@Auth : ZQB
@Time : 2024/10/25 2:06 PM
@File :susy.py
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
from .base_dataset import BaseDataset
import os


class HIGGS(BaseDataset):
    def __init__(self, cfg, rank, num_clients, num_models, dataset_mode):
        super(HIGGS, self).__init__(cfg, rank, num_clients, num_models, dataset_mode)
        self._csv_path = cfg.path

        self._get_train_aux_test_data_tensor()

    def _load_data_from_csv(self):
        data = pd.read_csv(self._csv_path, header=None)
        # data = data.sample(150000)
        data = data.sample(60000)
        # data = data.drop('id', axis=1)

        y = data.iloc[:, -1]
        x = data.iloc[:, :-1]

        # x = (x - x.min()) / (x.max() - x.min())
        # print(x)

        # One hot label
        enc = OneHotEncoder()
        y = enc.fit_transform(y.values.reshape(-1, 1)).toarray()
        y = y.astype(int)

        return x, y
