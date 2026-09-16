# -*- coding: utf-8 -*-
"""
@Auth : ZQB
@Time : 2024/12/16 8:03 PM
@File :covertype.py
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
from .base_dataset import BaseDataset
import time
import numpy as np


class Covertype(BaseDataset):
    def __init__(self, cfg, rank, num_clients, num_models, dataset_mode):
        super(Covertype, self).__init__(cfg, rank, num_clients, num_models, dataset_mode)
        self._csv_path = cfg.path

        self._get_train_aux_test_data_tensor()

    def _load_data_from_csv(self):
        data = pd.read_csv(self._csv_path, header=None)
        # sample_num = round(data.shape[0] * 0.8)
        # data = data.sample(sample_num)

        y = data.iloc[:, -1]
        x = data.iloc[:, :-1]

        # # Add noise to 10% of the data
        # x = x.astype(float)
        # num_samples = x.shape[0]
        # num_noisy = int(0.1 * num_samples)
        # noisy_indices = x.sample(n=num_noisy, random_state=42).index
        #
        # noise = pd.DataFrame(
        #     0.1 * np.random.randn(len(noisy_indices), x.shape[1]),
        #     index=noisy_indices,
        #     columns=x.columns
        # )
        # x.loc[noisy_indices] += noise

        # One hot label
        enc = OneHotEncoder()
        y = enc.fit_transform(y.values.reshape(-1, 1)).toarray()
        y = y.astype(int)

        return x, y