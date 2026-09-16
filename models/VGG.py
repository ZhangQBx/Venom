import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_convs: int = 2):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(num_convs):
            layers += [
                nn.Conv2d(ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VGGBottomModel(nn.Module):
    """
    VGG-style bottom model matching ResBottomModel style:
      - conv blocks (no residual)
      - AdaptiveAvgPool2d(1) + Linear to o_f
      - single-sample (C,H,W) support via unsqueeze + optional squeeze back

    Args:
        in_channels: input channels
        block_cfg: list of tuples (out_channels, num_convs, do_pool)
        o_f: output embedding dimension
        squeeze_single: mimic ResBottomModel returning [o_f] for a single sample
    """
    def __init__(
        self,
        in_channels: int,
        o_f: int,
        block_cfg=None,
        squeeze_single: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.o_f = o_f
        self.squeeze_single = squeeze_single

        # A compact VGG-like config (works well on CIFAR-sized inputs).
        # You can increase channels/depth if needed.
        if block_cfg is None:
            # (out_channels, num_convs, do_pool)
            block_cfg = [
                (64,  2, True),
                (128, 2, True),
                (256, 3, True),
            ]

        layers = []
        ch = in_channels
        for out_ch, num_convs, do_pool in block_cfg:
            layers.append(VGGBlock(ch, out_ch, num_convs=num_convs))
            if do_pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            ch = out_ch

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch, o_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single = True

        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if self.squeeze_single and single:
            x = x.squeeze(0)
        return x
