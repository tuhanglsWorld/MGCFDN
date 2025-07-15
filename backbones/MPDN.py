import torch
import torch.nn as nn
from timm.models.layers import DropPath
import torch.nn.functional as F


class PreprocessingBlock(nn.Module):
    def __init__(self, in_channels):
        super(PreprocessingBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=(9, 9), padding=(9 // 2, 9 // 2)),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

    def forward(self, input_x):
        out = self.net(input_x)
        return out


class TrasitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TrasitionBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

    def forward(self, input_x):
        return self.net(input_x)


class FeatureExtractorUnit(nn.Module):
    def __init__(self, in_channels, kernel_size=(5, 7), dilation=1):
        super(FeatureExtractorUnit, self).__init__()
        if dilation > 1:
            self.net_1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=(1, 1)),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.Conv2d(in_channels=24, out_channels=6, kernel_size=(kernel_size[0], kernel_size[0]),
                          padding=(kernel_size[0] // 2)),
                nn.BatchNorm2d(6),
                nn.ReLU(),
            )
            self.net_2 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=(1, 1)),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.Conv2d(in_channels=24, out_channels=6, kernel_size=(kernel_size[1], kernel_size[1]),
                          padding=(kernel_size[1] // 2)),
                nn.BatchNorm2d(6),
                nn.ReLU(),
            )
        else:
            self.net_1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=(1, 1)),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.Conv2d(in_channels=24, out_channels=6, kernel_size=(kernel_size[0], kernel_size[0]),
                          padding=(kernel_size[0] // 2) + dilation - 1, dilation=dilation),
                nn.BatchNorm2d(6),
                nn.ReLU(),
            )
            self.net_2 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=(1, 1)),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.Conv2d(in_channels=24, out_channels=6, kernel_size=(kernel_size[1], kernel_size[1]),
                          padding=(kernel_size[1] // 2) + dilation - 1, dilation=dilation),
                nn.BatchNorm2d(6),
                nn.ReLU(),
            )

    def forward(self, input_x):
        out_1 = self.net_1(input_x)
        out_2 = self.net_2(input_x)
        return torch.cat([out_1, out_2], dim=1)


class FeatureExtractorBlock_1(nn.Module):
    def __init__(self, ):
        super(FeatureExtractorBlock_1, self).__init__()
        self.net_1 = FeatureExtractorUnit(in_channels=24, kernel_size=(5, 7))
        self.net_2 = FeatureExtractorUnit(in_channels=36, kernel_size=(5, 7))
        self.net_3 = FeatureExtractorUnit(in_channels=48, kernel_size=(5, 7))
        self.net_4 = FeatureExtractorUnit(in_channels=60, kernel_size=(5, 7), dilation=3)

    def forward(self, input_x):
        out = input_x  # 24
        out_1 = self.net_1(out)  # 12
        out_2 = self.net_2(torch.cat([out, out_1], dim=1))  # 12
        out_3 = self.net_3(torch.cat([out_2, out_1, out], dim=1))  # 12
        out_4 = self.net_4(torch.cat([out_3, out_2, out_1, out], dim=1))  # 12
        return torch.cat([out_4, out_3, out_2, out_1, out], dim=1)


class FeatureExtractorBlock_2(nn.Module):
    def __init__(self):
        super(FeatureExtractorBlock_2, self).__init__()
        self.net_1 = FeatureExtractorUnit(in_channels=72, kernel_size=(3, 5))
        self.net_2 = FeatureExtractorUnit(in_channels=84, kernel_size=(3, 5))
        self.net_3 = FeatureExtractorUnit(in_channels=96, kernel_size=(3, 5))
        self.net_4 = FeatureExtractorUnit(in_channels=108, kernel_size=(3, 5))
        self.net_5 = FeatureExtractorUnit(in_channels=120, kernel_size=(3, 5), dilation=3)

    def forward(self, input_x):
        out = input_x  # 36
        out_1 = self.net_1(out)  # 12
        out_2 = self.net_2(torch.cat([out, out_1], dim=1))  # 12
        out_3 = self.net_3(torch.cat([out_2, out_1, out], dim=1))  # 12
        out_4 = self.net_4(torch.cat([out_3, out_2, out_1, out], dim=1))  # 12
        out_5 = self.net_5(torch.cat([out_4, out_3, out_2, out_1, out], dim=1))  # 12
        return torch.cat([out_5, out_4, out_3, out_2, out_1, out], dim=1)


class FeatureExtractorBlock_3(nn.Module):
    def __init__(self):
        super(FeatureExtractorBlock_3, self).__init__()
        self.net_1 = FeatureExtractorUnit(in_channels=132, kernel_size=(1, 3))
        self.net_2 = FeatureExtractorUnit(in_channels=144, kernel_size=(1, 3))
        self.net_3 = FeatureExtractorUnit(in_channels=156, kernel_size=(1, 3))
        self.net_4 = FeatureExtractorUnit(in_channels=168, kernel_size=(1, 3))
        self.net_5 = FeatureExtractorUnit(in_channels=180, kernel_size=(1, 3))
        self.net_6 = FeatureExtractorUnit(in_channels=192, kernel_size=(1, 3), dilation=3)

    def forward(self, input_x):
        out = input_x  # 48
        out_1 = self.net_1(out)  # 12
        out_2 = self.net_2(torch.cat([out, out_1], dim=1))  # 12
        out_3 = self.net_3(torch.cat([out_2, out_1, out], dim=1))  # 12
        out_4 = self.net_4(torch.cat([out_3, out_2, out_1, out], dim=1))  # 12
        out_5 = self.net_5(torch.cat([out_4, out_3, out_2, out_1, out], dim=1))  # 12
        out_6 = self.net_6(torch.cat([out_5, out_4, out_3, out_2, out_1, out], dim=1))  # 12
        return torch.cat([out_6, out_5, out_4, out_3, out_2, out_1, out], dim=1)


class MPDN(nn.Module):
    def __init__(self, in_channels=3, inner_channels=12):
        super(MPDN, self).__init__()
        self.preprocessingBlock = PreprocessingBlock(in_channels)
        self.featureExtractorBlock_1_left = FeatureExtractorBlock_1()
        self.featureExtractorBlock_1_right = FeatureExtractorBlock_1()
        self.trasitionBlock_1 = TrasitionBlock(in_channels=inner_channels * 12, out_channels=inner_channels * 6)
        self.featureExtractorBlock_2_left = FeatureExtractorBlock_2()
        self.featureExtractorBlock_2_right = FeatureExtractorBlock_2()
        self.trasitionBlock_2 = TrasitionBlock(in_channels=inner_channels * 22, out_channels=inner_channels * 11)
        self.featureExtractorBlock_3_left = FeatureExtractorBlock_3()
        self.featureExtractorBlock_3_right = FeatureExtractorBlock_3()
        self.trasitionBlock_3 = TrasitionBlock(in_channels=inner_channels * 34, out_channels=inner_channels * 17)

        # self.to_64_to_16 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
        # self.to_32_to_16 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, input_x, out_size=(16, 16)):
        out_1 = self.preprocessingBlock(input_x)  # 1 24 128 128
        out_2_left = self.featureExtractorBlock_1_left(out_1)  # 1 72 128 128
        out_2_right = self.featureExtractorBlock_1_right(out_1)  # 1 72 128 128
        out_3 = self.trasitionBlock_1(torch.cat([out_2_left, out_2_right], dim=1))  # 1 72 64 64
        out_4_left = self.featureExtractorBlock_2_left(out_3)  # 1 132 64 64
        out_4_right = self.featureExtractorBlock_2_right(out_3)  # 1 132 64 64
        out_5 = self.trasitionBlock_2(torch.cat([out_4_left, out_4_right], dim=1))  # 1 132 32 32
        out_6_left = self.featureExtractorBlock_3_left(out_5)  # 1 204 32 32
        out_6_right = self.featureExtractorBlock_3_right(out_5)  # 1 204 32 32
        out_7 = self.trasitionBlock_3(torch.cat([out_6_left, out_6_right], dim=1))  # 1 204 16 16

        x1_u = F.interpolate(
            out_3, size=out_size, mode='bilinear', align_corners=True)

        x2_u = F.interpolate(
            out_5, size=out_size, mode='bilinear', align_corners=True)

        x3_u = F.interpolate(
            out_7, size=out_size, mode='bilinear', align_corners=True)

        # return x3
        #x_all = torch.cat([self.to_64_to_16(out_3), self.to_32_to_16(out_5), out_7], dim=-3)
        x_all = torch.cat([x1_u, x2_u, x3_u], dim=-3)
        return x_all



if __name__ == '__main__':
    model = MPDN()
    result = torch.randn(size=(8, 3, 256, 256))
    x = model(result)
    print(x.size())
    total = sum([param.nelement() for param in model.parameters()])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
