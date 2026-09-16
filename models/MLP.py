# -*- coding: utf-8 -*-
"""
@Auth : ZQB
@Time : 2023/5/21 17:46
@File :MLP.py
"""
import torch
from torch import nn


class MLPBottomModel(nn.Module):
    def __init__(self, i_f, o_f, layers):
        super().__init__()
        self.in_dim = i_f
        self.out_dim = o_f
        self.layers = layers
        self.width = 20
        if layers == 1:
            self.dense_1 = nn.Linear(i_f, o_f)
            nn.init.xavier_normal_(self.dense_1.weight)
        elif layers == 2:
            self.dense_1 = nn.Linear(i_f, self.width)
            nn.init.xavier_normal_(self.dense_1.weight)
            self.dense_2 = nn.Linear(self.width, o_f)
            nn.init.xavier_normal_(self.dense_2.weight)
        elif layers == 3:
            self.dense_1 = nn.Linear(i_f, self.width)
            nn.init.xavier_normal_(self.dense_1.weight)
            self.dense_2 = nn.Linear(self.width, self.width)
            nn.init.xavier_normal_(self.dense_2.weight)
            self.dense_3 = nn.Linear(self.width, o_f)
            nn.init.xavier_normal_(self.dense_3.weight)
        elif layers == 4:
            self.dense_1 = nn.Linear(i_f, self.width)
            nn.init.xavier_normal_(self.dense_1.weight)
            self.dense_2 = nn.Linear(self.width, self.width)
            nn.init.xavier_normal_(self.dense_2.weight)
            self.dense_3 = nn.Linear(self.width, self.width)
            nn.init.xavier_normal_(self.dense_3.weight)
            self.dense_4 = nn.Linear(self.width, o_f)
            nn.init.xavier_normal_(self.dense_4.weight)
        elif layers == 5:
            self.dense_1 = nn.Linear(i_f, self.width)
            nn.init.xavier_normal_(self.dense_1.weight)
            self.dense_2 = nn.Linear(self.width, self.width)
            nn.init.xavier_normal_(self.dense_2.weight)
            self.dense_3 = nn.Linear(self.width, self.width)
            nn.init.xavier_normal_(self.dense_3.weight)
            self.dense_4 = nn.Linear(self.width, self.width)
            nn.init.xavier_normal_(self.dense_4.weight)
            self.dense_5 = nn.Linear(self.width, o_f)
            nn.init.xavier_normal_(self.dense_5.weight)
        else:
            raise ValueError("Invalid layers number")

    def forward(self, x):
        if self.layers == 1:
            x = self.dense_1(x)
        elif self.layers == 2:
            x = self.dense_1(x)
            x = torch.relu(x)
            x = self.dense_2(x)
        elif self.layers == 3:
            x = self.dense_1(x)
            x = torch.relu(x)
            x = self.dense_2(x)
            x = torch.relu(x)
            x = self.dense_3(x)
        elif self.layers == 4:
            x = self.dense_1(x)
            x = torch.relu(x)
            x = self.dense_2(x)
            x = torch.relu(x)
            x = self.dense_3(x)
            x = torch.relu(x)
            x = self.dense_4(x)
        elif self.layers == 5:
            x = self.dense_1(x)
            x = torch.relu(x)
            x = self.dense_2(x)
            x = torch.relu(x)
            x = self.dense_3(x)
            x = torch.relu(x)
            x = self.dense_4(x)
            x = torch.relu(x)
            x = self.dense_5(x)
        return x


class MLPTopModel(nn.Module):
    def __init__(self, i_f, o_f):
        super().__init__()
        self.width = 20
        self.dense_1 = nn.Linear(i_f, self.width, bias=False)
        nn.init.xavier_normal_(self.dense_1.weight)
        self.dense_2 = nn.Linear(self.width, self.width, bias=False)
        nn.init.xavier_normal_(self.dense_2.weight)
        self.dense_3 = nn.Linear(self.width, o_f, bias=False)
        nn.init.xavier_normal_(self.dense_3.weight)

        self.fc = nn.Linear(i_f, o_f)

    def forward(self, x):
        x = self.dense_1(x)
        x = torch.relu(x)
        x = self.dense_2(x)
        x = torch.relu(x)
        x = self.dense_3(x)
        # x = x.squeeze()
        # x = torch.relu(x)
        # x = self.fc(x)
        return x


class MLP(nn.Module):
    def __init__(self, n_f, o_f):
        super().__init__()
        self.dense_1 = nn.Linear(n_f, 20)
        nn.init.xavier_normal_(self.dense_1.weight)
        self.dense_2 = nn.Linear(20, o_f)
        nn.init.xavier_normal_(self.dense_2.weight)

    def forward(self, x):
        x = self.dense_1(x)
        x = torch.relu(x)
        x = self.dense_2(x)
        return x
