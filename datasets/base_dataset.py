# -*- coding: utf-8 -*-
"""
@Auth : ZQB
@Time : 2024/9/28 2:35 PM
@File :base_dataset.py
"""
import random
import time
import os
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, SVHN, CIFAR100, ImageFolder
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
import torch.utils.data as tud
from sklearn.preprocessing import StandardScaler, LabelBinarizer, MinMaxScaler


class BaseDataset(tud.Dataset):
    def __init__(self, cfg, rank, num_clients, num_models, dataset_mode):
        seed = cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        super(BaseDataset, self).__init__()

        self.cfg = cfg
        self.dataset_mode = dataset_mode
        self.train_rate = 0.7
        self.aux_rate = 0.1
        self.test_rate = 0.2

        self.num_clients = num_clients
        self.num_models = num_models
        self.rank = rank
        self.num_features = None

        self.num_train_data = None
        self.num_aux_data = None
        self.num_test_data = None

        self.train_data_np = None
        self.train_data_ts = None
        self.train_label_np = None
        self.train_label_ts = None

        self.aux_data_np = None
        self.aux_data_ts = None
        self.aux_label_np = None
        self.aux_label_ts = None
        self.replace_aux_with_noise = cfg.replace_aux_with_noise

        self.test_data_np = None
        self.test_data_ts = None
        self.test_label_np = None
        self.test_label_ts = None

        self.train_hash_res = None
        self.aux_hash_res = None
        self.test_hash_res = None

    def __len__(self):
        if self.dataset_mode == 1:
            return len(self.train_data_ts)
        elif self.dataset_mode == 2:
            return len(self.aux_data_ts)
        else:
            return len(self.test_data_ts)

    def __getitem__(self, index):
        if self.dataset_mode == 1:
            return self.train_data_ts[index], self.train_hash_res[index], self.train_label_ts[index]
        elif self.dataset_mode == 2:
            return self.aux_data_ts[index], self.aux_hash_res[index], self.aux_label_ts[index]
        else:
            return self.test_data_ts[index],self.test_hash_res[index], self.test_label_ts[index]

    # def reset_dataset_mode(self, mode):
    #     self.dataset_mode = mode
    #     print(self.dataset_mode)

    def _replace_aux_data_with_noise(self):
        if self.aux_data_np is None:
            print("aux_data_np is None, can't replace")
            return

        original_shape = self.aux_data_np.shape

        random_noise_np = np.random.randn(*original_shape)
        self.aux_data_np = random_noise_np
        self.aux_data_ts = torch.from_numpy(random_noise_np).float()

        print("replace aux data with noise")


    def _replace_aux_data_with_mnist(self, num_samples):
        # 1. Define the MNIST transforms
        mnist_transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize to 32x32
            transforms.Grayscale(num_output_channels=3),  # Convert 1 channel to 3 channels
            transforms.ToTensor(),  # Convert to a [C, H, W] tensor in [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

        try:
            mnist_trainset = MNIST(root='./data', train=True, download=True, transform=mnist_transform)
        except Exception as e:
            print(f"Error downloading MNIST: {e}. Aborting replacement.")
            return


        indices = np.random.choice(len(mnist_trainset), num_samples, replace=True)
        mnist_subset = Subset(mnist_trainset, indices)

        temp_loader = DataLoader(mnist_subset, batch_size=num_samples, shuffle=False)

        try:
            mnist_batch_data, _ = next(iter(temp_loader))
        except Exception as e:
            print(f"Error loading MNIST data with DataLoader: {e}")
            return

        return mnist_batch_data.numpy()

    def _replace_aux_data_with_svhn(self, num_samples):
        """
        (Helper function for ablation experiments)
        Load, transform, and return `num_samples` SVHN images.
        """
        print(f"Loading {num_samples} OOD samples from SVHN...")

        # 1. Define the SVHN transforms
        svhn_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to a [C, H, W] tensor in [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

        # 2. Load the SVHN dataset
        try:
            svhn_trainset = SVHN(root='./data', split='train', download=True, transform=svhn_transform)
        except Exception as e:
            print(f"Error downloading SVHN: {e}. Aborting replacement.")
            return None

        # 3. Randomly select `num_samples` samples
        if num_samples > len(svhn_trainset):
            print(f"Warning: Requested {num_samples}, using replacement.")
            indices = np.random.choice(len(svhn_trainset), num_samples, replace=True)
        else:
            indices = np.random.choice(len(svhn_trainset), num_samples, replace=False)
        svhn_subset = Subset(svhn_trainset, indices)

        # 4. Load the selected samples efficiently with DataLoader
        temp_loader = DataLoader(svhn_subset, batch_size=num_samples, shuffle=False)

        try:
            svhn_batch_data, _ = next(iter(temp_loader))
        except Exception as e:
            print(f"Error loading SVHN data with DataLoader: {e}")
            return None

        # print(f"Successfully loaded {svhn_batch_data.shape} SVHN data.")
        # 5. Return a NumPy array
        return svhn_batch_data.numpy()


    def _replace_aux_data_with_cifar100(self, num_samples):
        cifar100_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to a [C, H, W] tensor in [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

        # 2. Load the CIFAR-100 dataset
        try:
            cifar100_trainset = CIFAR100(root='./data', train=True, download=True, transform=cifar100_transform)
        except Exception as e:
            print(f"Error downloading CIFAR-100: {e}. Aborting replacement.")
            return

        # 3. Randomly select `num_samples` samples
        indices = np.random.choice(len(cifar100_trainset), num_samples, replace=True)
        cifar100_subset = Subset(cifar100_trainset, indices)

        # 4. Load the selected samples efficiently with DataLoader
        temp_loader = DataLoader(cifar100_subset, batch_size=num_samples, shuffle=False)

        try:
            cifar100_batch_data, _ = next(iter(temp_loader))
        except Exception as e:
            print(f"Error loading CIFAR-100 data with DataLoader: {e}")
            return

        # 5. Replace aux_data_np and aux_data_ts
        # self.aux_data_np = cifar100_batch_data.numpy()
        # self.aux_data_ts = torch.from_numpy(self.aux_data_np).float()

        return cifar100_batch_data.numpy()

        # print(f"Successfully replaced aux data with {self.aux_data_np.shape} CIFAR-100 data.")

    def _replace_aux_data_with_tiny_imagenet(self, num_samples):
        """
        (Helper function for ablation experiments)
        Load, transform, and return `num_samples` Tiny ImageNet images.
        This provides a suitable cross-domain test.
        *** Requirement: Extract tiny-imagenet-200.zip into ./data/ first. ***
        """
        print(f"Loading {num_samples} OOD samples from Tiny ImageNet...")
        print("Transforming Tiny ImageNet: Resizing to (32, 32) and Normalizing...")

        # Resize Tiny ImageNet images from 64x64 to 32x32
        tiny_imagenet_transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize to 32x32
            transforms.ToTensor(),  # Convert to a [C, H, W] tensor in [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

        # Point ImageFolder to the directory containing class folders, such as 'train'
        tiny_imagenet_path = os.path.join('../datasets/image', 'tiny-imagenet-200', 'train')

        try:
            # Load with ImageFolder
            dataset = ImageFolder(root=tiny_imagenet_path, transform=tiny_imagenet_transform)
            if len(dataset) == 0:
                raise FileNotFoundError  # Handle an empty directory
        except FileNotFoundError:
            print(f"Error: Tiny ImageNet data NOT FOUND at '{tiny_imagenet_path}'.")
            print("Please download 'tiny-imagenet-200.zip', unzip it, and place it in the './data/' directory.")
            return None
        except Exception as e:
            print(f"Error loading Tiny ImageNet: {e}. Aborting replacement.")
            return None

        # 3. Randomly select `num_samples` samples
        if num_samples > len(dataset):
            print(f"Warning: Requested {num_samples}, using replacement.")
            indices = np.random.choice(len(dataset), num_samples, replace=True)
        else:
            indices = np.random.choice(len(dataset), num_samples, replace=False)

        subset = Subset(dataset, indices)

        # 4. Load the selected samples efficiently with DataLoader
        temp_loader = DataLoader(subset, batch_size=num_samples, shuffle=False)

        try:
            batch_data, _ = next(iter(temp_loader))
        except Exception as e:
            print(f"Error loading Tiny ImageNet data with DataLoader: {e}")
            return None

        print(f"Successfully loaded {batch_data.shape} Tiny ImageNet data.")
        # 5. Return a NumPy array
        return batch_data.numpy()


    def _get_train_aux_test_data_tensor(self):
        if self.cfg.dataset == "nuswide":
            pass
        else:
            x, y = self._get_all_data_np()
            if self.rank == self.num_clients:
                print(f"{self.rank}, Init Server Dataset")
            else:
                x = self._generate_vertical_numpy_data(x)
            if self.dataset_mode == 1:
                self.train_data_np = x[:self.num_train_data]
                self.train_data_ts = torch.from_numpy(x[:self.num_train_data]).float()
                self.train_label_np = y[:self.num_train_data]
                self.train_label_ts = torch.from_numpy(y[:self.num_train_data]).float()
            elif self.dataset_mode == 2:
                self.aux_data_np = x[self.num_train_data: self.num_train_data + self.num_aux_data]
                self.aux_data_ts = torch.from_numpy(x[self.num_train_data: self.num_train_data + self.num_aux_data]).float()
                self.aux_label_np = y[self.num_train_data: self.num_train_data + self.num_aux_data]
                self.aux_label_ts = torch.from_numpy(y[self.num_train_data: self.num_train_data + self.num_aux_data]).float()
                if self.replace_aux_with_noise:
                    self._replace_aux_data_with_noise()
            else:
                self.test_data_np = x[self.num_train_data + self.num_aux_data:]
                self.test_data_ts = torch.from_numpy(x[self.num_train_data + self.num_aux_data:]).float()
                self.test_label_np = y[self.num_train_data + self.num_aux_data:]
                self.test_label_ts = torch.from_numpy(y[self.num_train_data + self.num_aux_data:]).float()

        if self.num_models == 1 or self.rank == self.num_clients:
            if self.dataset_mode == 1:
                self.train_hash_res = [0 for _ in range(self.num_train_data)]
            elif self.dataset_mode == 2:
                self.aux_hash_res = [0 for _ in range(self.num_aux_data)]
            else:
                self.test_hash_res = [0 for _ in range(self.num_test_data)]
            return
        # num_feature = self.train_data_np.shape[1]
        # num_feature = x.shape[1]
        lsh = LSH(1, self.num_features, self.rank, self.num_models)
        if self.dataset_mode == 1:
            self.train_hash_res = lsh.hash(self.train_data_np, self.dataset_mode)
            assert len(self.train_hash_res) == self.num_train_data
        elif self.dataset_mode == 2:
            self.aux_hash_res = lsh.hash(self.aux_data_np, self.dataset_mode)
            assert len(self.aux_hash_res) == self.num_aux_data
        else:
            self.test_hash_res = lsh.hash(self.test_data_np, self.dataset_mode)
            assert len(self.test_hash_res) == self.num_test_data


        # assert len(self.train_hash_res) == self.num_train_data
        # assert len(self.aux_hash_res) == self.num_aux_data
        # assert len(self.test_hash_res) == self.num_test_data

    def _get_train_aux_test_image_tensor(self):
        x_train, y_train, x_aux, y_aux, x_test, y_test = self._get_all_image()
        if self.dataset_mode == 1:
            self.train_data_np = self._generate_vertical_image_data(x_train)
            self.train_data_ts = torch.from_numpy(self.train_data_np).float()
            self.train_label_np = y_train
            self.train_label_ts = torch.from_numpy(self.train_label_np).long()
        elif self.dataset_mode == 2:
            if self.replace_aux_with_noise:
                num_samples = len(x_aux)
                # x_aux = self._replace_aux_data_with_cifar100(num_samples)
                # x_aux = self._replace_aux_data_with_tiny_imagenet(num_samples)
                x_aux = self._replace_aux_data_with_mnist(num_samples)
                # x_aux = self._replace_aux_data_with_svhn(num_samples)
            self.aux_data_np = self._generate_vertical_image_data(x_aux)
            self.aux_data_ts = torch.from_numpy(self.aux_data_np).float()
            self.aux_label_np = y_aux
            self.aux_label_ts = torch.from_numpy(self.aux_label_np).long()
            # if self.replace_aux_with_noise:
            #     cifar100_data = self._replace_aux_data_with_cifar100()
        else:
            self.test_data_np = self._generate_vertical_image_data(x_test)
            self.test_data_ts = torch.from_numpy(self.test_data_np).float()
            self.test_label_np = y_test
            self.test_label_ts = torch.from_numpy(self.test_label_np).long()
            # print(x_train.shape, x_aux.shape, x_test.shape)

        if self.num_models == 1 or self.rank == self.num_clients:
            self.train_hash_res = [0 for _ in range(self.num_train_data)]
            self.aux_hash_res = [0 for _ in range(self.num_aux_data)]
            self.test_hash_res = [0 for _ in range(self.num_test_data)]
            return


        flatten = self._flatten_image(mode=self.dataset_mode)
        num_feature = flatten.shape[1]

        lsh = LSH(1, num_feature, self.rank, self.num_models)
        # self.train_hash_res, self.aux_hash_res, self.test_hash_res = lsh.hash(train_flatten, aux_flatten, test_flatten)
        if self.dataset_mode == 1:
            self.train_hash_res = lsh.hash(flatten, self.dataset_mode)
            assert len(self.train_hash_res) == self.num_train_data
        elif self.dataset_mode == 2:
            self.aux_hash_res = lsh.hash(flatten, self.dataset_mode)
            assert len(self.aux_hash_res) == self.num_aux_data
        else:
            self.test_hash_res = lsh.hash(flatten, self.dataset_mode)
            assert len(self.test_hash_res) == self.num_test_data

        # print(self.train_hash_res[:10])

        # assert len(self.train_hash_res) == self.num_train_data
        # assert len(self.aux_hash_res) == self.num_aux_data
        # assert len(self.test_hash_res) == self.num_test_data


    def _generate_vertical_image_data(self, data):
        num_last_dimension = data.shape[-1]
        num_split_features = num_last_dimension // self.num_clients
        data_feature_indexes = [i for i in range(num_last_dimension)]
        data_feature_index_begin = self.rank * num_split_features
        index = data_feature_indexes[data_feature_index_begin:data_feature_index_begin + num_split_features]

        if self.rank == self.num_clients - 1:
            index = data_feature_indexes[data_feature_index_begin:]
            client_data = data[:, :, :, data_feature_index_begin:]
        else:
            client_data = data[:, :, :, data_feature_index_begin:data_feature_index_begin + num_split_features]
        self.num_features = len(index)

        return client_data

    # def _flatten_image(self):
    #     train_flatten = []
    #     aux_flatten = []
    #     test_flatten = []
    #
    #     for data in self.train_data_np:
    #         train_flatten.append(data.flatten())
    #     for data in self.aux_data_np:
    #         aux_flatten.append(data.flatten())
    #     for data in self.test_data_np:
    #         test_flatten.append(data.flatten())
    #
    #     train_flatten = np.array(train_flatten)
    #     aux_flatten = np.array(aux_flatten)
    #     test_flatten = np.array(test_flatten)
    #
    #     return train_flatten, aux_flatten, test_flatten

    def _flatten_image(self, mode):
        flatten = []
        if mode == 1:
            for data in self.train_data_np:
                flatten.append(data.flatten())
        elif mode == 2:
            for data in self.aux_data_np:
                flatten.append(data.flatten())
        else:
            for data in self.test_data_np:
                flatten.append(data.flatten())

        flatten = np.array(flatten)

        return flatten


    def _generate_vertical_numpy_data(self, data):
        num_data_features = data.shape[1]
        num_split_features = num_data_features // self.num_clients
        data_feature_indexes = [index for index in range(num_data_features)]
        random.shuffle(data_feature_indexes)

        data_feature_index_begin = self.rank * num_split_features

        index = data_feature_indexes[data_feature_index_begin:data_feature_index_begin + num_split_features]
        if self.rank == self.num_clients - 1:
            index = data_feature_indexes[data_feature_index_begin:]
            client_data = self._generate_client_specific_numpy_data(index, data)
        else:
            client_data = self._generate_client_specific_numpy_data(index, data)
        print(index)
        self.num_features = len(index)

        # if self.rank == 0:
        #     index = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 20, 21, 49]
        #     client_data = self._generate_client_specific_numpy_data(index, data)
        # elif self.rank == 1:
        #     index = [8, 14, 15, 16, 17, 24, 25, 26, 27, 28, 50, 51, 22]
        #     client_data = self._generate_client_specific_numpy_data(index, data)
        # elif self.rank == 2:
        #     index = [9, 18, 19, 29, 30, 31, 32, 33, 34, 35, 36, 52, 23]
        #     client_data = self._generate_client_specific_numpy_data(index, data)
        # elif self.rank == 3:
        #     index = [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 53, 48]
        #     client_data = self._generate_client_specific_numpy_data(index, data)
        #
        # self.num_features = len(index)

        return client_data

    def _generate_client_specific_numpy_data(self, feature_indexes, data):
        client_data = []
        for data_item in data:
            client_data_item = [data_item[feature_index] for feature_index in feature_indexes]
            client_data.append(client_data_item)

        return np.asarray(client_data)

    def _get_all_data_np(self):
        x, y = self._load_data_from_csv()
        # print("scale")
        scalar = MinMaxScaler()
        x = scalar.fit_transform(x)
        x = np.asarray(x)
        y = np.asarray(y)

        # x_min = np.min(x, axis=0)
        # x_max = np.max(x, axis=0)
        #
        # x = (x - x_min) / (x_max - x_min)
        # print(x)

        # y = self._label_binarizer(y)
        x, y = self._shuffle_data(x, y)
        # print("shuffle")
        self.num_train_data = int(len(x) * self.train_rate)
        self.num_aux_data = int(len(x) * self.aux_rate)
        self.num_test_data = len(x) - self.num_train_data - self.num_aux_data

        return x, y

    def _get_all_image(self):
        x, y = self._load_image()
        x, y = self._shuffle_data(x, y)

        self.num_train_data = int(len(x) * self.train_rate)
        self.num_aux_data = int(len(x) * self.aux_rate)
        self.num_test_data = len(x) - self.num_train_data - self.num_aux_data

        train_end_idx = self.num_train_data
        aux_end_idx = self.num_train_data + self.num_aux_data

        x_train, y_train = x[:train_end_idx], y[:train_end_idx]
        x_aux, y_aux = x[train_end_idx:aux_end_idx], y[train_end_idx:aux_end_idx]
        x_test, y_test = x[aux_end_idx:], y[aux_end_idx:]

        # print(f"Image dataset split: Train={len(y_train)}, Aux={len(y_aux)}, Test={len(y_test)}")
        assert len(y_train) == self.num_train_data
        assert len(y_aux) == self.num_aux_data
        assert len(y_test) == self.num_test_data

        return x_train, y_train, x_aux, y_aux, x_test, y_test

        # x_aux = x_train[-5000:, :, :, :]
        # y_aux = y_train[-5000:]
        # x_train = x_train[:-5000, :, :, :]
        # y_train = y_train[:-5000]
        # self.num_train_data = len(y_train)
        # self.num_aux_data = len(y_aux)
        # self.num_test_data = len(y_test)
        #
        # # print(self.num_train_data, self.num_aux_data, self.num_test_data)
        #
        # return x_train, y_train, x_aux, y_aux, x_test, y_test

    @staticmethod
    def _shuffle_data(x, y):
        index = [i for i in range(len(x))]
        random.shuffle(index)
        x = x[index]
        y = y[index]
        # print("index: ", index[:10])

        return x, y

    @staticmethod
    def _label_binarizer(y):
        label_as_one_hot = LabelBinarizer()
        y = label_as_one_hot.fit_transform(y)

        return y

    def _load_data_from_csv(self):
        raise NotImplementedError

    def _load_image(self):
        raise NotImplementedError

class LSH:
    def __init__(self, tables_num, num_feature, rank, num_bottom_models):
        self.seed = np.random.randint(0, 1000, 10)
        np.random.seed(self.seed[rank])
        self.R = np.random.randn(num_feature, tables_num)
        # np.random.seed(origin_seed)
        self.num_bottom_models = num_bottom_models
        # self.hash_tables = [dict() for _ in range(tables_num)]

    def hash(self, data, mode):
        if self.num_bottom_models == 1:
            hash_res = np.zeros(len(data))
        elif self.num_bottom_models == 2:
            lsh_signature = np.sign(np.matmul(data, self.R)) / 2 + 0.5
            hash_res = lsh_signature.astype(np.int32).flatten()
        else:
            pass

        final_hash_res = np.array([random.randint(0, self.num_bottom_models - 1) for _ in hash_res])
        return final_hash_res

        # return train_hash_res, aux_hash_res, test_hash_res

    def _get_split_standard(self, result):
        sorted_data_hash_res = np.sort(result.flatten())
        standard = [np.NINF]
        step = len(sorted_data_hash_res) // self.num_bottom_models
        for i in range(self.num_bottom_models - 1):
            standard.append(sorted_data_hash_res[(i + 1) * step])

        standard.append(np.PINF)
        # print("standard: ", standard)

        return standard

    def _get_final_hash_res(self, hash_res, standard):
        # hash_res = lsh.hash(data_np)
        # split_datasets = []
        # split_datasets_length = []
        new_hash_res = np.array([None for _ in range(len(hash_res))])

        for i in range(len(standard) - 1):
            pos = np.where((hash_res > standard[i]) & (hash_res <= standard[i + 1]))[0]
            new_hash_res[pos] = i

        return new_hash_res
