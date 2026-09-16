import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import MyDatasetTS
from utils.defense import *


class Client:
    def __init__(self, cfg, rank, client_info_dict, device, defense = None):
        self.cfg = cfg
        self.rank = rank

        self.train_dataset = client_info_dict["train_dataset"]
        self.train_datasets = []
        self.test_dataset = client_info_dict["test_dataset"]
        self.test_datasets = []
        self.aux_dataset = client_info_dict["aux_dataset"]
        self.aux_datasets = []

        self.bottom_models = client_info_dict["bottom_model"]
        self.surrogate_model = client_info_dict["surrogate_model"]
        self.surrogate_models = []
        # self.surrogate_model = None
        self.generator = client_info_dict["generator"]
        self.defense = defense
        self.device = device
        self.mask = None

        self.bottom_optimizers = []
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=cfg.bs)
        self.train_dataloaders = []
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=cfg.bs)
        self.test_dataloaders = []
        self.aux_dataloader = DataLoader(self.aux_dataset, batch_size=cfg.bs)
        self.aux_dataloaders = []

        self.train_batch = len(self.train_dataloader)
        self.test_batch = len(self.test_dataloader)
        self.aux_batch = len(self.aux_dataloader)

        self.embeddings_out = None
        self.hash_res = None
        self.bottom_models_embeddings_list = []
        self.projection_matrix = torch.randn(cfg.bottom_model_output_dim, cfg.bottom_model_output_dim).to(self.device)

        for model in self.bottom_models:
            self.bottom_optimizers.append(torch.optim.Adam(model.parameters(), lr=cfg.lr))
            # self.bottom_optimizers.append()

        for i in range(len(self.bottom_models)):
            datas = []
            labels = []
            for data, hash_res, label in self.train_dataloader:
                mask = (hash_res == i)
                datas.append(data[mask].numpy())
                labels.append(label[mask].numpy())

            datas = np.concatenate(datas)
            datas = torch.from_numpy(datas)
            labels = np.concatenate(labels)
            labels = torch.from_numpy(labels)

            bottom_model_dataset = MyDatasetTS(datas, labels)

            print(f"Client {self.rank} train dataset {i} size: {len(bottom_model_dataset)}")

            self.train_datasets.append(bottom_model_dataset)
            self.train_dataloaders.append(DataLoader(bottom_model_dataset, self.cfg.bs))

        for i in range(len(self.bottom_models)):
            datas = []
            labels = []
            for data, hash_res, label in self.test_dataloader:
                mask = (hash_res == i)
                datas.append(data[mask].numpy())
                labels.append(label[mask].numpy())

            datas = np.concatenate(datas)
            datas = torch.from_numpy(datas)
            labels = np.concatenate(labels)
            labels = torch.from_numpy(labels)

            bottom_model_dataset = MyDatasetTS(datas, labels)

            print(f"Client {self.rank} test dataset {i} size: {len(bottom_model_dataset)}")

            self.test_datasets.append(bottom_model_dataset)
            self.test_dataloaders.append(DataLoader(bottom_model_dataset, self.cfg.bs))

        for i in range(len(self.bottom_models)):
            datas = []
            labels = []
            for data, hash_res, label in self.aux_dataloader:
                mask = (hash_res == i)
                datas.append(data[mask].numpy())
                labels.append(label[mask].numpy())

            datas = np.concatenate(datas)
            datas = torch.from_numpy(datas)
            labels = np.concatenate(labels)
            labels = torch.from_numpy(labels)

            bottom_model_dataset = MyDatasetTS(datas, labels)

            print(f"Client {self.rank} aux dataset {i} size: {len(bottom_model_dataset)}")

            self.aux_datasets.append(bottom_model_dataset)
            self.aux_dataloaders.append(DataLoader(bottom_model_dataset, self.cfg.bs))

    def get_embeddings(self, mode, batch_index):
        """
        let the bottom models generate embeddings_out, which is defensed
        :param mode: 1 for train, 2 for test
        :param batch_index: the index of batch
        :return: embeddings_out: embeddings_out of the batch (tensor)
        """
        if mode == 1:
            dataloader = self.train_dataloader
        elif mode == 2:
            dataloader = self.test_dataloader
        else:
            dataloader = self.aux_dataloader

        for model in self.bottom_models:
            model.to(self.device)
            model.train()

        data = None
        hash_res = None
        index = 0
        for index, (data, hash_res, _) in enumerate(dataloader):
            if index == batch_index:
                break

        if data is None:
            raise ValueError("fetch data failed")

        if index < batch_index:
            return None

        self.hash_res = hash_res
        data = data.to(self.device)

        self.embeddings_out = torch.zeros(len(data), self.cfg.bottom_model_output_dim).to(self.device)
        self.bottom_models_embeddings_list = []

        for i in range(len(self.bottom_models)):
            mask = (self.hash_res == i)
            embedding = self.bottom_models[i](data[mask])
            self.bottom_models_embeddings_list.append(embedding)

        for i in range(len(self.bottom_models)):
            mask = (self.hash_res == i)
            self.embeddings_out[mask] = self.bottom_models_embeddings_list[i]

        self.embeddings_out.retain_grad()

        if self.defense == "noisy":
            self.embeddings_out = noisy_embedding_based_defense(self.embeddings_out, self.cfg.noise_std)
        elif self.defense == "pruning":
            self.embeddings_out, self.mask = pruning_embedding_based_defense_by_removing_elements(self.embeddings_out, self.cfg.pruning_ratio)

        embeddings = self.embeddings_out.clone().cpu().detach().numpy()

        return embeddings

    def update_bottom_models(self, grads, batch):
        grads = grads.to(self.device)

        if self.defense == "pruning":
            grads = torch.matmul(grads, self.mask)

        grads_list = []
        for i in range(len(self.bottom_models)):
            mask = (self.hash_res == i)
            grads_list.append(grads[mask])

        for i in range(len(self.bottom_optimizers)):
            # print("\n", i, "\t", grads_list[i])
            self.bottom_optimizers[i].zero_grad()
            self.bottom_models_embeddings_list[i].backward(grads_list[i])
            self.bottom_optimizers[i].step()

    def train_surrogate_model(self, aux_emb_dataset):
        """
        train the surrogate model
        :param aux_emb_dataset: the dataset of the embeddings_out generated by the bottom models
        :return: None
        """
        self.surrogate_model.to(self.device)
        self.surrogate_model.train()

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.surrogate_model.parameters(), lr=self.cfg.lr_ms)
        aux_emb_dataloader = DataLoader(aux_emb_dataset, batch_size=self.cfg.bs_ms)

        for epoch in tqdm(range(self.cfg.epochs), desc=f"Surrogate Model Train-Client {self.rank}", dynamic_ncols=True):
            epoch_loss = 0
            # right_train = 0
            for x, y in aux_emb_dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.surrogate_model(x)
                loss = criterion(output, y)
                epoch_loss += loss.item()

                # right_train += output.argmax(dim=1).eq(y.argmax(dim=1)).sum().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            tqdm.write(f">>>Epoch: {epoch}, Loss: {epoch_loss}")


    def get_surrogate_model_output(self, batch):
        """
        get the output of the surrogate model
        :param batch: the embeddings_out generated by the bottom models
        :return: the output of the surrogate model
        """
        self.surrogate_model.to(self.device)
        self.surrogate_model.train()

        dataloader = self.test_dataloader

        i = 0
        data = None
        for i, (data, _, _) in enumerate(dataloader):
            if i == batch:
                break

        if i < batch:
            return None

        data = data.to(self.device)

        return self.surrogate_model(data).cpu().detach().numpy()
