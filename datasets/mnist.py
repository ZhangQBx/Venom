# -*- coding: utf-8 -*-
"""
@Auth : ZQB
@Time : 2024/12/18 3:00 PM
@File :mnist.py
"""
import os
import torch
from .base_dataset import BaseDataset
from torchvision import datasets, transforms
import time


class MNIST(BaseDataset):
    def __init__(self, cfg, rank, num_clients, num_models, dataset_mode):
        super(MNIST, self).__init__(cfg, rank, num_clients, num_models, dataset_mode)
        # self._image_path = os.getcwd() + cfg.path
        self._image_path = cfg.path

        self._get_train_aux_test_image_tensor()

    # def _load_image(self):
    #     data_tf = transforms.Compose(
    #         [transforms.ToTensor(),
    #          transforms.Normalize((0.1307,), (0.3081,))]
    #     )
    #     # print(self._image_path)
    #     # time.sleep(100000)
    #
    #     train_dataset = datasets.MNIST(root=self._image_path, train=True,
    #                                    transform=data_tf,  download=False)
    #     test_dataset = datasets.MNIST(root=self._image_path, train=False,
    #                                   transform=data_tf, download=False)
    #
    #     train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    #     test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    #
    #     x_train = None
    #     y_train = None
    #     for images, labels in train_loader:
    #         x_train = images
    #         y_train = labels
    #         x_train = x_train.numpy()
    #         y_train = y_train.numpy()
    #
    #     x_test = None
    #     y_test = None
    #     for images, labels in test_loader:
    #         x_test = images
    #         y_test = labels
    #         x_test = x_test.numpy()
    #         y_test = y_test.numpy()
    #
    #     return x_train, y_train, x_test, y_test

    def _load_image(self):
        train_dataset = datasets.MNIST(root=self._image_path,
                                       train=True, download=False)
        test_dataset = datasets.MNIST(root=self._image_path,
                                      train=False, download=False)

        x_train, y_train = train_dataset.data, train_dataset.targets
        x_test, y_test = test_dataset.data, test_dataset.targets
        x = torch.cat([x_train, x_test], dim=0)
        y = torch.cat([y_train, y_test], dim=0)
        x = x.float() / 255.0
        x = x.unsqueeze(1)

        normalize = transforms.Normalize((0.1307,), (0.3081,))
        x = normalize(x)
        # print(x.shape)
        # time.sleep(10000)
        return x.numpy(), y.numpy()