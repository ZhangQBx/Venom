import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t=6):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(
                in_channels * t,
                in_channels * t,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels * t,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        out = self.residual(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            out = out + x
        return out


class MobileNetV2BottomModel(nn.Module):
    """
    MobileNetV2-style bottom model aligned with your ResBottomModel style:
      - accepts (in_channels, o_f)
      - uses AdaptiveAvgPool2d(1) then a Linear layer to output embedding of size o_f
      - supports single-sample input (C,H,W) by unsqueezing and (optionally) squeezing back

    Notes:
      - This is derived from your provided MobileNetV2 implementation, but removes the classification conv2
        and replaces it with a Linear head, matching ResBottomModel's `avgpool + fc`.
    """
    def __init__(self, in_channels: int, o_f: int, width_mult: float = 1.0, squeeze_single: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.o_f = o_f
        self.squeeze_single = squeeze_single

        # Base channels (following common MobileNetV2: 32, 16, 24, 32, 64, 96, 160, 320, 1280)
        c32 = int(32 * width_mult)
        c16 = int(16 * width_mult)
        c24 = int(24 * width_mult)
        c32_ = int(32 * width_mult)
        c64 = int(64 * width_mult)
        c96 = int(96 * width_mult)
        c160 = int(160 * width_mult)
        c320 = int(320 * width_mult)
        c1280 = int(1280 * width_mult)

        # "pre" block (match style: conv + bn + relu)
        # Your original code used kernel_size=1, padding=1, which is unusual and changes spatial size.
        # Here we use the standard MobileNetV2 stem: 3x3, stride=1, padding=1.
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, c32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c32),
            nn.ReLU6(inplace=True),
        )

        # Stages (same topology as your code)
        self.stage1 = LinearBottleNeck(c32, c16, stride=1, t=1)
        self.stage2 = self._make_stage(repeat=2, in_channels=c16, out_channels=c24, stride=2, t=6)
        self.stage3 = self._make_stage(repeat=3, in_channels=c24, out_channels=c32_, stride=2, t=6)
        self.stage4 = self._make_stage(repeat=4, in_channels=c32_, out_channels=c64, stride=2, t=6)
        self.stage5 = self._make_stage(repeat=3, in_channels=c64, out_channels=c96, stride=1, t=6)
        self.stage6 = self._make_stage(repeat=3, in_channels=c96, out_channels=c160, stride=1, t=6)
        self.stage7 = LinearBottleNeck(c160, c320, stride=1, t=6)

        # Final 1x1 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(c320, c1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1280),
            nn.ReLU6(inplace=True),
        )

        # Match ResBottomModel: avgpool + fc to o_f
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c1280, o_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single = True

        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if self.squeeze_single and single:
            x = x.squeeze(0)
        return x

    @staticmethod
    def _make_stage(repeat, in_channels, out_channels, stride, t):
        layers = [LinearBottleNeck(in_channels, out_channels, stride, t)]
        for _ in range(repeat - 1):
            layers.append(LinearBottleNeck(out_channels, out_channels, stride=1, t=t))
        return nn.Sequential(*layers)


def mobilenetv2_bottom(in_channels: int, o_f: int, **kwargs):
    return MobileNetV2BottomModel(in_channels=in_channels, o_f=o_f, **kwargs)
