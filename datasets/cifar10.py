import os
import torch
from .base_dataset import BaseDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time


class CIFAR10(BaseDataset):
    def __init__(self, cfg, rank, num_clients, num_models, dataset_mode):
        super(CIFAR10, self).__init__(cfg, rank, num_clients, num_models, dataset_mode)
        # self._image_path = os.getcwd() + cfg.path
        self._image_path = cfg.path

        self._get_train_aux_test_image_tensor()

    # def _load_image(self):
    #     """
    #     Load CIFAR-10 dataset and convert it to numpy arrays.
    #     """
    #     # For CIFAR-10, images are 3-channel, so normalization is different.
    #     # data_tf = transforms.Compose(
    #     #     [transforms.ToTensor(),
    #     #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    #     # )
    #
    #     data_tf =transforms.Compose([transforms.RandomCrop(32, padding=4),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
    #                                  ])
    #
    #     # Load CIFAR-10 training and test datasets
    #     train_dataset = datasets.CIFAR10(root=self._image_path, train=True,
    #                                      transform=data_tf, download=False)
    #     test_dataset = datasets.CIFAR10(root=self._image_path, train=False,
    #                                     transform=data_tf, download=False)
    #
    #     # Use DataLoader to load all data into memory at once
    #     train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    #     test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    #
    #     x_train = None
    #     y_train = None
    #     # Extract training data and labels
    #     for images, labels in train_loader:
    #         x_train = images
    #         y_train = labels
    #         x_train = x_train.numpy()
    #         y_train = y_train.numpy()
    #
    #     x_test = None
    #     y_test = None
    #     # Extract test data and labels
    #     for images, labels in test_loader:
    #         x_test = images
    #         y_test = labels
    #         x_test = x_test.numpy()
    #         y_test = y_test.numpy()
    #
    #     return x_train, y_train, x_test, y_test

    def _load_image(self):
        """
        [Improved version]
        Efficiently load CIFAR-10, apply transforms, and combine the training and test sets.

        Iterate over the dataset directly to apply transforms, including augmentation,
        without DataLoader overhead, then return the combined x and y NumPy arrays.
        """
        # Define data transforms, including augmentation
        # Test sets usually omit random augmentation; share transforms here to preserve the original behavior
        data_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Standard deviations are usually not 1
        ])

        # 1. Load dataset objects with the transform
        train_dataset = datasets.CIFAR10(root=self._image_path, train=True,
                                         transform=data_tf, download=False)
        test_dataset = datasets.CIFAR10(root=self._image_path, train=False,
                                        transform=data_tf, download=False)

        # 2. Apply transforms and collect data efficiently
        # Direct iteration avoids DataLoader batching and multiprocessing overhead
        # Use list comprehensions to collect the data efficiently
        x_train = torch.stack([img for img, _ in train_dataset])
        y_train = torch.tensor(train_dataset.targets)

        x_test = torch.stack([img for img, _ in test_dataset])
        y_test = torch.tensor(test_dataset.targets)

        # 3. Combine training and test sets
        x = torch.cat([x_train, x_test], dim=0)
        y = torch.cat([y_train, y_test], dim=0)

        # 4. Convert PyTorch tensors to NumPy arrays and return them
        return x.numpy(), y.numpy()