import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
from .base_dataset import BaseDataset

class Diabetes(BaseDataset):
    def __init__(self, cfg, rank, num_clients, num_models, dataset_mode):
        super(Diabetes, self).__init__(cfg, rank, num_clients, num_models, dataset_mode)
        self._csv_path = cfg.path

        self._get_train_aux_test_data_tensor()

    def _load_data_from_csv(self):
        data = pd.read_csv(self._csv_path)
        data = pd.get_dummies(data, columns=['gender'], drop_first=True)
        smoking_freq = data['smoking_history'].value_counts(normalize=True).to_dict()
        data['smoking_encoded'] = data['smoking_history'].map(smoking_freq)
        bool_cols = data.select_dtypes(include='bool').columns
        data[bool_cols] = data[bool_cols].astype(int)
        data = data.drop("smoking_history", axis=1)

        x = data.drop("diabetes", axis=1)
        y = data["diabetes"]

        # One hot label
        enc = OneHotEncoder()
        y = enc.fit_transform(y.values.reshape(-1, 1)).toarray()
        y = y.astype(int)

        return x, y