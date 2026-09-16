import os
import random
import time
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
from .base_dataset import BaseDataset

class NUSWIDE(BaseDataset):
    def __init__(self, cfg, rank, num_clients, num_models, dataset_mode):
        super(NUSWIDE, self).__init__(cfg, rank, num_clients, num_models, dataset_mode)
        self._multimodal_path = cfg.path
        self.selected_label = ['buildings', 'grass', 'animal', 'water', 'person']

        x_image_train, x_text_train, y_train = self._get_labeled_data(None, 'Train')
        x_image_test, x_text_test, y_test = self._get_labeled_data(None, 'Test')
        x_image_all = np.concatenate((x_image_train, x_image_test), axis=0)
        x_text_all = np.concatenate((x_text_train, x_text_test), axis=0)
        y_all = np.concatenate((y_train, y_test), axis=0)
        x_image_all, x_text_all, y_all = self._shuffle_multimodal_data(x_image_all, x_text_all, y_all)
        enc = OneHotEncoder()
        y_all = enc.fit_transform(y_all.reshape(-1, 1)).toarray()
        y_all = y_all.astype(int)

        num_samples = len(y_all)
        self.num_train_data = int(num_samples * self.train_rate)
        self.num_aux_data = int(num_samples * self.aux_rate)
        self.num_test_data = num_samples - self.num_train_data - self.num_aux_data
        train_end_idx = self.num_train_data
        aux_end_idx = self.num_train_data + self.num_aux_data

        x_all_local = self._assign_modality_to_client(x_image_all, x_text_all)
        self.num_features = x_all_local.shape[1]

        if self.dataset_mode == 1:
            self.train_data_np = x_all_local[:train_end_idx]
            self.train_data_ts = torch.from_numpy(x_all_local[:train_end_idx]).float()
            self.train_label_np = y_all[:train_end_idx]
            self.train_label_ts = torch.from_numpy(y_all[:train_end_idx]).float()
            self.num_features = self.train_data_np.shape[1]
        elif self.dataset_mode == 2:
            self.aux_data_np = x_all_local[train_end_idx:aux_end_idx]
            self.aux_data_ts = torch.from_numpy(x_all_local[train_end_idx:aux_end_idx]).float()
            self.aux_label_np = y_all[train_end_idx:aux_end_idx]
            self.aux_label_ts = torch.from_numpy(y_all[train_end_idx:aux_end_idx]).float()
            self.num_features = self.aux_data_np.shape[1]
        elif self.dataset_mode == 3:
            self.test_data_np = x_all_local[aux_end_idx:]
            self.test_data_ts = torch.from_numpy(x_all_local[aux_end_idx:]).float()
            self.test_label_np = y_all[aux_end_idx:]
            self.test_label_ts = torch.from_numpy(y_all[aux_end_idx:]).float()
            self.num_features = self.test_data_np.shape[1]

        self._get_train_aux_test_data_tensor()

    def _get_labeled_data(self, n_samples, dtype="Train"):
        data_path = "Groundtruth/TrainTestLabels/"
        dfs = []
        for label in self.selected_label:
            file = os.path.join(self._multimodal_path, data_path, "_".join(["Labels", label, dtype]) + ".txt")
            # print("Loading {}.".format(file))
            df = pd.read_csv(file, header=None, engine="c")
            df.columns = [label]
            dfs.append(df)
        data_labels = pd.concat(dfs, axis=1)
        if len(self.selected_label) > 1:
            selected = data_labels[data_labels.sum(axis=1) == 1]
        else:
            selected = data_labels
        # get XA, which are image low level features
        features_path = "Low_Level_Features"
        dfs = []
        for file in os.listdir(os.path.join(self._multimodal_path, features_path)):
            if file.startswith("_".join([dtype, "Normalized"])):
                # print("Loading {}.".format(os.path.join(self._multimodal_path, features_path, file)))
                df = pd.read_csv(os.path.join(self._multimodal_path, features_path, file), header=None, sep=" ", engine="c")
                df.dropna(axis=1, inplace=True)
                dfs.append(df)
        data_XA = pd.concat(dfs, axis=1)
        data_X_image_selected = data_XA.loc[selected.index]
        # get XB, which are tags
        tag_path = "NUS_WID_Tags/"
        file = "_".join([dtype, "Tags1k"]) + ".dat"
        # print("Loading {}.".format(file))
        tagsdf = pd.read_csv(os.path.join(self._multimodal_path, tag_path, file), header=None, sep="\t", engine="c")
        tagsdf.dropna(axis=1, inplace=True)
        data_X_text_selected = tagsdf.loc[selected.index]

        if n_samples is None:
            return data_X_image_selected.values[:], data_X_text_selected.values[:], np.argmax(selected.values[:], 1)

        return data_X_image_selected.values[:n_samples], data_X_text_selected.values[:n_samples], np.argmax(
            selected.values[:n_samples])


    def _assign_modality_to_client(self, x_image_all, x_text_all):
        if self.rank == 0:
            print(f"Client {self.rank}: assigned IMAGE.")
            return x_image_all
        else:  # rank == 1
            print(f"Client {self.rank}: assigned TEXT.")
            return x_text_all


    @staticmethod
    def _shuffle_multimodal_data(x_image, x_text, y):
        index = [i for i in range(len(y))]
        random.shuffle(index)
        return x_image[index], x_text[index], y[index]