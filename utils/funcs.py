# -*- coding: utf-8 -*-
"""
@Auth : ZQB
@Time : 2024/9/29 11:30 AM
@File :funcs.py
"""
import time
import random

import numpy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional
from torch.utils.data import Dataset
import torch.nn.functional as F
from datasets import dataset
from datasets.base_dataset import LSH
from models import *
from models.CNN import CNNTopModel, CNNBottomModel, ResBottomModel, LocalModelForCifar10
from models.MobileNetV2 import MobileNetV2BottomModel
from models.VGG import VGGBottomModel


def init_server(cfg, dataset_name, top_input_dim, num_clients, num_models):
    train = dataset(dataset_name, cfg, 0, num_clients, num_models, 1)
    # time.sleep(1000000000)
    aux = dataset(dataset_name, cfg, 0, num_clients, num_models, 2)
    test = dataset(dataset_name, cfg, 0, num_clients, num_models, 3)

    # print(train.train_label_np.shape)

    # label_dim = train.train_label_np.shape[1]
    # top_model = MLPTopModel(top_input_dim, label_dim)

    if cfg.type == "tabular":
        label_dim = train.train_label_np.shape[1]
        top_model = MLPTopModel(top_input_dim, label_dim)
        # top_model = MLPTopModel(top_input_dim, 1)
    elif cfg.type == "image":
        top_model = CNNTopModel(top_input_dim, 10, layers=cfg.get("top_model_layers", 4))
    else:
        top_model = None

    info_dict = {"train_dataset": train, "aux_dataset": aux, "test_dataset": test, "top_model": top_model}

    return info_dict


def init_client(cfg, dataset_name, rank, num_clients, num_models):
    train_dataset = dataset(dataset_name, cfg, rank, num_clients, num_models, 1)
    aux_dataset = dataset(dataset_name, cfg, rank, num_clients, num_models, 2)
    test_dataset = dataset(dataset_name, cfg, rank, num_clients, num_models, 3)

    input_dim = train_dataset.num_features

    # print(train_dataset[:2])
    # time.sleep(10000)

    bottom_model = []
    for i in range(num_models):
        if cfg.type == "tabular":
            input_dim = train_dataset.train_data_np.shape[1]
            model = MLPBottomModel(input_dim, cfg.bottom_model_output_dim, cfg.bottom_model_layers)
        elif cfg.type == "image":
            if cfg.dataset == "mnist":
                model = ResBottomModel(1, cfg.bottom_model_layers, cfg.bottom_model_output_dim)
            elif cfg.dataset == "cifar10":
                model = ResBottomModel(3, cfg.bottom_model_layers, cfg.bottom_model_output_dim)
            elif cfg.dataset == "tiny_imagenet":
                model = ResBottomModel(3, cfg.bottom_model_layers, cfg.bottom_model_output_dim)
            else:
                model = None
            # model = CNNBottomModel(input_dim, cfg.bottom_model_output_dim)
            # model = ResBottomModel(1, cfg.bottom_model_output_dim)
            # model = LocalModelForCifar10()
            # model = CNNBottomModel(input_dim, cfg.bottom_model_output_dim)
        else:
            model = None
        # model = MLPBottomModel(input_dim, cfg.bottom_model_output_dim, cfg.bottom_model_layers)
        bottom_model.append(model)

    # surrogate_model = MLPBottomModel(input_dim, cfg.bottom_model_output_dim, 3)

    if cfg.type == "tabular":
        surrogate_model = MLPBottomModel(input_dim, cfg.bottom_model_output_dim, 3)
        generator = Generator(input_dim, input_dim)
    elif cfg.type == "image":
        if cfg.dataset == "mnist":
            surrogate_model = ResBottomModel(1, cfg.surrogate_model_layers, cfg.bottom_model_output_dim)
        elif cfg.dataset == "cifar10":
            surrogate_model = ResBottomModel(3, cfg.surrogate_model_layers, cfg.bottom_model_output_dim)
        elif cfg.dataset == "tiny_imagenet":
            surrogate_model = ResBottomModel(3, cfg.surrogate_model_layers, cfg.bottom_model_output_dim)
            # print("MobileNetV2")
            # surrogate_model = MobileNetV2BottomModel(3, cfg.bottom_model_output_dim)
            # print("VGG")
            # surrogate_model = VGGBottomModel(3, cfg.bottom_model_output_dim)
    else:
        surrogate_model = None

    generator = Generator(input_dim, input_dim)

    if cfg.defense == "pruning":
        pruning_ratio = cfg.pruning_ratio
        top_input_dim = round(cfg.bottom_model_output_dim * (1 - pruning_ratio))
        surrogate_model = MLPBottomModel(input_dim, top_input_dim, cfg.bottom_model_layers)

    info_dict = {"train_dataset": train_dataset, "aux_dataset": aux_dataset, "test_dataset": test_dataset,
                 "bottom_model": bottom_model, "surrogate_model": surrogate_model, "generator": generator}

    return info_dict


def l2_norm_loss(bottom_embedding, aux_embedding):
    return torch.norm(bottom_embedding - aux_embedding, dim=1)


def l2_norm_loss_1(bottom_embedding, aux_embedding):
    return torch.norm(bottom_embedding - aux_embedding)


def kldiv_loss(bottom_embedding, aux_embedding):
    return nn.KLDivLoss(reduction='batchmean') \
        (F.log_softmax(bottom_embedding, dim=1), F.softmax(aux_embedding, dim=1))


def gradient_matching_loss(b, a):
    cos_sim = F.cosine_similarity(b.grad, a.grad, dim=1)
    cos_sim = torch.sum(cos_sim)

    return cos_sim


def all_loss(l2, kld, sim, alpha, beta, theta):
    return alpha * l2 + beta * kld + theta * sim


def adjust_learning_rate(epoch, lr, lr_gamma, lr_step, aux_optimizer):
    if epoch in lr_step:
        lr *= lr_gamma
        for param_group in aux_optimizer.param_groups:
            param_group['lr'] = lr

        print(f"Learning rate decay: {lr}")

    return lr


# class LSH:
#     def __init__(self, tables_num, num_feature, rank, num_bottom_models):
#         self.seed = np.random.randint(0, 1000, 4)
#         np.random.seed(self.seed[rank])
#         self.R = np.random.randn(num_feature, tables_num)
#         # np.random.seed(origin_seed)
#         self.num_bottom_models = num_bottom_models
#         # self.hash_tables = [dict() for _ in range(tables_num)]
#
#     def hash(self, train, aux, test):
#         if self.num_bottom_models == 2:
#             train_lsh_signature = np.sign(np.matmul(train, self.R)) / 2 + 0.5
#             train_hash_res = train_lsh_signature.astype(np.int32).flatten()
#             aux_lsh_signature = np.sign(np.matmul(aux, self.R)) / 2 + 0.5
#             aux_hash_res = aux_lsh_signature.astype(np.int32).flatten()
#             test_lsh_signature = np.sign(np.matmul(test, self.R)) / 2 + 0.5
#             test_hash_res = test_lsh_signature.astype(np.int32).flatten()
#         else:
#             train_hash_res = np.matmul(train, self.R).reshape(-1)
#             standard = self._get_split_standard(train_hash_res)
#             train_hash_res = self._get_final_hash_res(train_hash_res, standard)
#             aux_hash_res = np.matmul(aux, self.R).reshape(-1)
#             aux_hash_res = self._get_final_hash_res(aux_hash_res, standard)
#             test_hash_res = np.matmul(test, self.R).reshape(-1)
#             test_hash_res = self._get_final_hash_res(test_hash_res, standard)
#         # print(train_hash_res[:20])
#         # time.sleep(100000)
#
#         return train_hash_res, aux_hash_res, test_hash_res
#
#     def _get_split_standard(self, result):
#         sorted_data_hash_res = np.sort(result.flatten())
#         standard = [np.NINF]
#         step = len(sorted_data_hash_res) // self.num_bottom_models
#         for i in range(self.num_bottom_models - 1):
#             standard.append(sorted_data_hash_res[(i + 1) * step])
#
#         standard.append(np.PINF)
#         print("standard: ", standard)
#
#         return standard
#
#     def _get_final_hash_res(self, hash_res, standard):
#         # hash_res = lsh.hash(data_np)
#         # split_datasets = []
#         # split_datasets_length = []
#         new_hash_res = np.array([None for _ in range(len(hash_res))])
#
#         for i in range(len(standard) - 1):
#             pos = np.where((hash_res > standard[i]) & (hash_res <= standard[i + 1]))[0]
#             new_hash_res[pos] = i
#
#         return new_hash_res


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.data = torch.from_numpy(x).float()
        self.label = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


class MyDatasetTS(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


class MyDatasetKnn_Near_Far(Dataset):
    def __init__(self, x, y, z):
        self.data = x
        self.near_knn = y
        self.far_knn = z
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index], self.near_knn[index], self.far_knn[index]


class MyDatasetKnn_Near_Far_WithIndex(Dataset):
    """
    Corrected dataset class that includes the current sample's index.
    Return original data, the current sample index, nearest-neighbor indices, and farthest-neighbor indices.
    """
    def __init__(self, aux_data, near_indices, far_indices):
        self.aux_data = aux_data
        self.near_indices = near_indices
        self.far_indices = far_indices

    def __len__(self):
        return len(self.aux_data)

    def __getitem__(self, index):
        # Return the data, its own index, and its neighbors' indices
        return self.aux_data[index], index, self.near_indices[index], self.far_indices[index]


# class MyDatasetKnn_Near_Far(Dataset):
#     """
#     A custom dataset returning:
#     1. Original data (aux_data)
#     2. Its original index (index)
#     3. Indices of K nearest neighbors (near_indices)
#     4. Indices of K farthest neighbors (far_indices)
#     """
#     def __init__(self, aux_data, near_indices, far_indices):
#         self.aux_data = aux_data
#         self.near_indices = near_indices
#         self.far_indices = far_indices
#
#     def __len__(self):
#         return len(self.aux_data)
#
#     def __getitem__(self, index):
#         # Return the data, its own index, and its neighbors' indices
#         return self.aux_data[index], index, self.near_indices[index], self.far_indices[index]



class MyDatasetWeightsTS(Dataset):
    def __init__(self, x, y, weights):
        self.data = x
        self.label = y
        self.weights = weights

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.weights[index]


class AttackDatasetTS(Dataset):
    def __init__(self, cfg, rank, x, y, num_surrogates, need_hash=False):
        seed = cfg.surrogate_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.data = x
        self.label = y

        self.x_np = x.numpy()
        self.y_np = y.numpy()

        # self.need_hash = need_hash
        self.hash_res = None

        if need_hash:
            # lsh = LSH(1, x.shape[1], rank, num_surrogates, cfg.surrogate_seed)
            # self.hash_res, _, _ = lsh.hash(self.x_np, self.x_np, self.x_np)
            self.hash_res = np.random.randint(0, num_surrogates, len(self.data))
        else:
            self.hash_res = np.zeros(len(self.data))

        # print(len(self.hash_res==0), len(self.hash_res==1))

        # time.sleep(100000)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.hash_res[index], self.label[index]


class BinaryLSH:
    def __init__(self, tables_num, num_feature, cid):
        self.seed = np.random.randint(0, 1000, 4)
        np.random.seed(self.seed[cid])
        # print(self.seed)
        self.R = np.random.randn(num_feature, tables_num)
        # np.random.seed(origin_seed)
        self.hash_tables = [dict() for _ in range(tables_num)]

    def hash(self, inputs):
        lsh_signature = np.sign(np.matmul(inputs, self.R)) / 2 + 0.5
        signatures = lsh_signature.astype(np.int32).flatten()

        return signatures

    def insert(self, inputs):
        inputs = np.array(inputs)
        if len(inputs.shape) == 1:
            inputs = inputs.reshape([1, -1])

        signatures = self.hash(inputs)

        return signatures


def compute_distance_matrix(X):
    """
    Compute the distance matrix of input tensor X.
    X has shape [batch_size, embedding_dim].
    """
    n = X.size(0)
    # Compute Euclidean distances between every pair of samples
    dists = torch.cdist(X, X, p=2)  # Compute L2 distances
    return dists


def center_distance_matrix(D):
    """
    Center the distance matrix D.
    """
    n = D.size(0)
    row_mean = D.mean(dim=1, keepdim=True)
    col_mean = D.mean(dim=0, keepdim=True)
    total_mean = D.mean()

    # Center the matrix
    D_centered = D - row_mean - col_mean + total_mean
    return D_centered


def distance_correlation(X, Y):
    """
    Compute the distance correlation (dCor) between X and Y.
    X and Y have shape [batch_size, embedding_dim].
    """
    # Compute the distance matrices of X and Y
    D_X = compute_distance_matrix(X)
    D_Y = compute_distance_matrix(Y)

    # Center the distance matrices
    D_X_centered = center_distance_matrix(D_X)
    D_Y_centered = center_distance_matrix(D_Y)

    # Compute covariance
    cov_dxy = (D_X_centered * D_Y_centered).sum() / (D_X.size(0) ** 2)
    cov_dxx = (D_X_centered * D_X_centered).sum() / (D_X.size(0) ** 2)
    cov_dyy = (D_Y_centered * D_Y_centered).sum() / (D_Y.size(0) ** 2)

    # Compute distance correlation
    dcor = torch.sqrt(cov_dxy / torch.sqrt(cov_dxx * cov_dyy))
    return dcor


# Define a loss that minimizes distance correlation
class DCorLoss(torch.nn.Module):
    def __init__(self):
        super(DCorLoss, self).__init__()

    def forward(self, model_1_output, model_2_output):
        """
        Inputs: model_1_output and model_2_output, each of shape [batch_size, embedding_dim].
        Output: The computed distance correlation loss.
        """
        return distance_correlation(model_1_output, model_2_output)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def total_variation_loss(x):
    return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))


def ssim(img1, img2, window_size=11, size_average=True, full=False):
    img1 = img1.float()
    img2 = img2.float()

    def gaussian(window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(window_size, sigma):
        _1d_window = gaussian(window_size, sigma).unsqueeze(1)
        window = _1d_window.mm(_1d_window.t()).unsqueeze(0).unsqueeze(0)
        return window

    window = create_window(window_size, 1.5).cuda()


    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=1)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=1) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if full:
        return ssim_map.mean(), ssim_map
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1)
