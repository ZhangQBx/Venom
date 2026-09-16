# -*- coding: utf-8 -*-
"""
@Auth : ZQB
@Time : 2024/10/21 12:11 PM
@File :Generator.py
"""
import time

import torch.nn as nn
from torch.nn import functional as F

class Generator(nn.Module):
    def __init__(self, i_f, o_f):
        super(Generator, self).__init__()
        self.fc = nn.Linear(i_f, o_f)
        self.fc_1 = nn.Linear(i_f, 20)
        # nn.init.xavier_normal_(self.fc_1.weight)
        # self.fc_2 = nn.Linear(40, 40)
        # nn.init.xavier_normal_(self.fc_2.weight)
        self.fc_2 = nn.Linear(20, o_f)
        # nn.init.xavier_normal_(self.fc_3.weight)

    def forward(self, x):
        # x = self.fc(x)
        # x = F.relu(x)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        # x = self.fc_3(x)
        # x = F.relu(x)
        # x = F.relu(x)
        return x

class ImageGenerator(nn.Module):
    def __init__(self, noise_dim, channels, img_w, img_h):
        super(ImageGenerator, self).__init__()
        self.img_w = img_w
        self.img_h = img_h
        self.channels = channels
        self.fc1 = nn.Linear(noise_dim, 256)
        self.fc2 = nn.Linear(256, channels * img_w * img_h)

        # self.image_height = img_h
        # self.image_width = img_w
        #
        # self.fc = nn.Linear(noise_dim, 256 * 7 * 7)
        #
        # self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        # self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # self.deconv3 = nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1)
        #
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.tanh(x)

        x = x.view(-1, self.channels, self.img_w, self.img_h)



        return x

    def crop_or_pad(self, x, target_height, target_width):
        current_height, current_width = x.shape[2], x.shape[3]

        if current_height > target_height:
            x = x[:, :, :target_height, :]
        elif current_height < target_height:
            pad_height = target_height - current_height
            x = nn.functional.pad(x, (0, 0, 0, pad_height))

        if current_width > target_width:
            x = x[:, :, :, :target_width]
        elif current_width < target_width:
            pad_width = target_width - current_width
            x = nn.functional.pad(x, (0, pad_width, 0, 0))

        return x