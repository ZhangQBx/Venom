import torch
from torch.utils.data import DataLoader

from models import MLPTopModel


class Server:
    def __init__(self, cfg, server_info_dict, device):
        self.cfg = cfg
        self.device = device
        # self.top_model.to(device)
        # self.top_model.train()
        self.train_dataset = server_info_dict["train_dataset"]
        self.test_dataset = server_info_dict["test_dataset"]
        self.aux_dataset = server_info_dict["aux_dataset"]
        self.top_model = server_info_dict["top_model"]

        self.train_loader = DataLoader(self.train_dataset, batch_size=cfg.bs)
        self.test_loader = DataLoader(self.test_dataset, batch_size=cfg.bs)
        self.aux_loader = DataLoader(self.aux_dataset, batch_size=cfg.bs)
        self.top_optimizer = torch.optim.Adam(self.top_model.parameters(), lr=cfg.lr)

    def train(self, embeddings: torch.Tensor, batch_index) -> (float, int, torch.Tensor):
        """
        train the top model
        :param embeddings: embeddings_out of the batch
        :param batch_index: the index of batch
        :return: new_grads: the gradients of the top model
        """
        self.top_model.to(self.device)
        self.top_model.train()

        label = None
        index = 0
        for index, (_, _, label) in enumerate(self.train_loader):
            if index == batch_index:
                break

        if label is None:
            raise ValueError("fetch data failed")

        if index < batch_index:
            return None

        label = label.to(self.device)
        embeddings = embeddings.to(self.device)

        embeddings.retain_grad()

        outputs = self.top_model(embeddings)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, label)
        right_train = outputs.argmax(dim=1).eq(label.argmax(dim=1)).sum().item()

        self.top_optimizer.zero_grad()
        loss.backward()
        self.top_optimizer.step()

        # print(embeddings.grad.size())

        return loss.item(), right_train, embeddings.grad

    def update_top_model(self):
        pass

    def get_predictions(self, embeddings, batch_index):
        """
        get the predictions of the top model
        :param embeddings: embeddings_out of the batch
        :param batch_index: the index of batch
        :return: predictions: the predictions of the top model
        """
        self.top_model.to(self.device)
        self.top_model.train()

        label = None
        index = 0
        for index, (_, _, label) in enumerate(self.test_loader):
            if index == batch_index:
                break

        if label is None:
            raise ValueError("fetch data failed")

        if index < batch_index:
            return None

        label = label.to(self.device)
        embeddings = embeddings.to(self.device)

        outputs = self.top_model(embeddings)
        _, predicted = outputs.max(1)
        predicted = predicted.cpu()
        # pred_list.append(predicted.numpy())
        label_ = torch.from_numpy(label.nonzero().cpu().numpy()[:, 1])
        right_pred = predicted.eq(label_).sum().item()

        return predicted, right_pred