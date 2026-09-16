# -*- coding: utf-8 -*-
"""
@Auth : ZQB
@Time : 2024/9/28 6:06 PM
@File :get_dataset.py
"""
from datasets.bank import Bank
from datasets.covertype import Covertype
from datasets.diabetes import Diabetes
from datasets.higgs import HIGGS
from datasets.houseprice import HousePrice
from datasets.mushroom import Mushroom
from datasets.rice import Rice
from datasets.susy import SUSY
from datasets.mnist import MNIST
from datasets.cifar10 import CIFAR10
from datasets.nuswide import NUSWIDE
from datasets.tiny_imagenet import TinyImageNet
import torch.utils.data as tud

from datasets.year import Year


def dataset(name, cfg, rank, num_clients, num_models, dataset_mode):
    """

    :param client_rank: client rank
    :param cfg: config file
    :param train: train or test
    :param is_label_owner: if this client hold the label
    :param comm_status: local or distribute
    :return: dataset object
    """
    dataset = None
    if name == 'bank':
        dataset = Bank(cfg, rank, num_clients, num_models, dataset_mode)
    if name == 'susy':
        dataset = SUSY(cfg, rank, num_clients, num_models, dataset_mode)
    if name == "mnist":
        dataset = MNIST(cfg, rank, num_clients, num_models, dataset_mode)
    if name == "cifar10":
        dataset = CIFAR10(cfg, rank, num_clients, num_models, dataset_mode)
    if name == "covertype":
        dataset = Covertype(cfg, rank, num_clients, num_models, dataset_mode)
    if name == "rice":
        dataset = Rice(cfg, rank, num_clients, num_models, dataset_mode)
    if name == "higgs":
        dataset = HIGGS(cfg, rank, num_clients, num_models, dataset_mode)
    if name == "mushroom":
        dataset = Mushroom(cfg, rank, num_clients, num_models, dataset_mode)
    if name == "houseprice":
        dataset = HousePrice(cfg, rank, num_clients, num_models, dataset_mode)
    if name == "year":
        dataset = Year(cfg, rank, num_clients, num_models, dataset_mode)
    if name == "diabetes":
        dataset = Diabetes(cfg, rank, num_clients, num_models, dataset_mode)
    if name == "nuswide":
        dataset = NUSWIDE(cfg, rank, num_clients, num_models, dataset_mode)
    if name == "tiny_imagenet":
        dataset = TinyImageNet(cfg, rank, num_clients, num_models, dataset_mode)
    if dataset is None:
        raise ValueError("No such dataset")


    return dataset


