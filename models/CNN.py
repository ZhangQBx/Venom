import time

from torch import nn
from torch.nn import functional as F
from torchvision import models

class LocalModelForCifar10(nn.Module):
    def __init__(self):
        super(LocalModelForCifar10, self).__init__()
        # self.args = args
        self.backbone = models.resnet18(pretrained=False)
        num_ftrs = self.backbone.fc.in_features
        # if self.args.client_num == 2:
        self.backbone.fc = nn.Linear(num_ftrs, 128)

    def forward(self, x):
        x = self.backbone(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResBottomModel(nn.Module):
    def __init__(self, in_channels, block_num, o_f):
        super(ResBottomModel, self).__init__()
        self.block_num = block_num
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        if block_num == 1:
            self.rb1 = ResidualBlock(64, 64)
        elif block_num == 2:
            self.rb1 = ResidualBlock(64, 64)
            self.rb2 = ResidualBlock(64, 64)
        elif block_num == 3:
            self.rb1 = ResidualBlock(64, 64)
            self.rb2 = ResidualBlock(64, 64)
            self.rb3 = ResidualBlock(64, 64)
        elif block_num == 4:
            self.rb1 = ResidualBlock(64, 64)
            self.rb2 = ResidualBlock(64, 64)
            self.rb3 = ResidualBlock(64, 64)
            self.rb4 = ResidualBlock(64, 64)
        elif block_num == 5:
            self.rb1 = ResidualBlock(64, 64)
            self.rb2 = ResidualBlock(64, 64)
            self.rb3 = ResidualBlock(64, 64)
            self.rb4 = ResidualBlock(64, 64)
            self.rb5 = ResidualBlock(64, 64)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, o_f)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        # print(x.shape)
        x = F.relu(self.bn(self.conv(x)))
        if self.block_num == 1:
            x = self.rb1(x)
        elif self.block_num == 2:
            x = self.rb1(x)
            x = self.rb2(x)
        elif self.block_num == 3:
            x = self.rb1(x)
            x = self.rb2(x)
            x = self.rb3(x)
        elif self.block_num == 4:
            x = self.rb1(x)
            x = self.rb2(x)
            x = self.rb3(x)
            x = self.rb4(x)
        elif self.block_num == 5:
            x = self.rb1(x)
            x = self.rb2(x)
            x = self.rb3(x)
            x = self.rb4(x)
            x = self.rb5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.squeeze(0)

        # time.sleep(1000000)
        return x


class CNNBottomModel(nn.Module):
    def __init__(self, num_features, o_f):
        super(CNNBottomModel, self).__init__()
        self.full_connect_dim1 = num_features // 4
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * self.full_connect_dim1, 128)
        self.fc2 = nn.Linear(128, o_f)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * self.full_connect_dim1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(0)
        return x


class CNNTopModel(nn.Module):
    def __init__(self, i_f, num_classes=10, layers=4):
        super(CNNTopModel, self).__init__()
        if layers not in (3, 4):
            raise ValueError("top_model_layers must be 3 or 4")
        self.layers = layers
        self.fc1 = nn.Linear(i_f, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, num_classes if layers == 3 else 20)
        if layers == 4:
            self.fc4 = nn.Linear(20, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        if self.layers == 4:
            x = F.relu(self.fc4(x))
        return x
