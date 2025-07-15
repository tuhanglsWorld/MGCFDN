import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import math
from backbones.VIG import DeepGCN

class FineGrainedModule(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(FineGrainedModule, self).__init__()
        self.conv_pre = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv_next = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_pre(x)
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        out = self.conv_next(out)
        return out


class MultiGranularityConsistencyModule(nn.Module):
    def __init__(self, in_channels=640, inner_channels=256, dropout=0.6, topk=40):
        super(MultiGranularityConsistencyModule, self).__init__()
        self.topk = topk
        self.fine_grained_module = FineGrainedModule(channel=topk)
        self.conv_select = nn.Sequential(
            nn.Conv2d(topk, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        self.aspp = torchvision.models.segmentation.deeplabv3.ASPP(in_channels=in_channels,
                                                                   out_channels=inner_channels,
                                                                   atrous_rates=[12, 24, 36])
        self.feature_drop = nn.Dropout2d(p=dropout)
        self.alpha = nn.Parameter(torch.tensor(5., dtype=torch.float32))

    def fn_gauss(self, x, u, s):
        return torch.exp(-(x - u) ** 2 / (2 * s ** 2))

    def gen_auto_corr_suppression_matrix(self, x_in, h, w, rat_s=0.1):
        sigma = h * rat_s, w * rat_s
        # c = h * w
        b, c, h2, w2 = x_in.shape
        ind_r, ind_c = torch.arange(h2).float(), torch.arange(w2).float()
        ind_r, ind_c = ind_r.view(1, 1, -1, 1).expand_as(x_in), ind_c.view(1, 1, 1, -1).expand_as(x_in)
        # center
        c_indices = torch.from_numpy(np.indices((h, w))).float()
        c_ind_r, c_ind_c = c_indices[0].reshape(-1), c_indices[1].reshape(-1)
        cent_r, cent_c = c_ind_r.reshape(1, c, 1, 1).expand_as(x_in), c_ind_c.reshape(1, c, 1, 1).expand_as(x_in)

        gaus_r, gaus_c = self.fn_gauss(ind_r, cent_r, sigma[0]), self.fn_gauss(ind_c, cent_c, sigma[1])
        out_g = 1 - gaus_r * gaus_c
        out_g = out_g.to(x_in.device)
        out = out_g * x_in
        return out

    def get_local_grained_info(self, x, f2):
        local_attention = self.conv_select(f2)
        f3 = self.feature_drop(self.aspp(x)) * local_attention
        return f3

    def get_global_grained_info(self, x, global_attention):
        global_attention = global_attention / (global_attention.sum(dim=-3, keepdim=True) + 1e-8)
        b, c, h2, w2 = x.shape
        b, _, h1, w1 = global_attention.shape
        f4 = self.feature_drop(
            torch.bmm(x.reshape(b, c, -1), global_attention.reshape(b, h2 * w2, h1 * w1)).reshape(b, c, h1, w1))
        return f4

    def forward(self, x):
        b, c, h1, w1 = x.shape
        h2, w2 = h1, w1
        fi = fj = F.normalize(x, p=2, dim=-3)
        # h1 * w1, h2 * w2 compute global autocorrelation information
        auto_global_attention = torch.matmul(fi.permute(0, 2, 3, 1).view(b, -1, c), fj.view(b, c, -1))
        # suppression of self-information by suppression matrix
        global_attention = self.gen_auto_corr_suppression_matrix(auto_global_attention.view(b, -1, h1, w1), h1, w1,
                                                                 rat_s=0.05).reshape(b, h1 * w1, h2 * w2)
        # quality perception enhancement
        global_attention = F.softmax(global_attention * self.alpha, dim=-1) * F.softmax(global_attention * self.alpha,
                                                                                        dim=-2)
        # shape change
        global_attention = global_attention.reshape(b, h1, w1, h2, w2).view(b, h1 * w1, h2, w2)
        # filter the first K pieces of information to get coarse-grained information
        f1, _ = torch.topk(global_attention, k=self.topk, dim=-3)
        # get fine-grained information
        f2 = self.fine_grained_module(f1)
        # local granularity information is obtained
        f3 = self.get_local_grained_info(x, f2)
        # get global granularity information
        f4 = self.get_global_grained_info(f3, global_attention)

        feature = torch.cat((f3, f4, f1, f2), dim=-3)
        return feature


class MultiGranularityConsistencyForgeryDetectionNet(nn.Module):
    def __init__(self, feature_channels=640, inner_channels=256, out_channel=1, dropout=0.6, topk=40):
        super(MultiGranularityConsistencyForgeryDetectionNet, self).__init__()

        self.visual_feature_extractor = DeepGCN()
        self.multi_granularity_consistency_module = MultiGranularityConsistencyModule(in_channels=feature_channels,
                                                                                      inner_channels=inner_channels,
                                                                                      dropout=dropout, topk=topk)
        self.decode_head = nn.Sequential(
            nn.Conv2d(inner_channels * 2 + topk * 2, inner_channels * 2, 1),
            nn.BatchNorm2d(inner_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(2 * inner_channels, inner_channels, 3, padding=1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inner_channels, out_channel, 1),
        )
        self.decode_head.apply(weights_init_normal)

    def forward(self, x, out_size=(256, 256)):
        x = self.visual_feature_extractor(x)
        multi_granularity_feature = self.multi_granularity_consistency_module(x)
        mask = F.interpolate(
            self.decode_head(multi_granularity_feature), size=out_size, mode='bilinear', align_corners=True)
        return mask


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


"""
Flops:  25.17 GMac
Params: 92.37 M
"""
from ptflops import get_model_complexity_info


def calc_flops_params():
    model = MultiGranularityConsistencyForgeryDetectionNet()
    result = torch.randn(size=(8, 3, 256, 256))
    pre_result = model(result)
    # print(pre_result.size())
    total = sum([param.nelement() for param in model.parameters()])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    print('Flops:  ' + flops)
    print('Params: ' + params)

    # flops, params = get_model_complexity_info(model.multi_granularity_consistency_module, (640, 40, 40),
    #                                           as_strings=True,
    #                                           print_per_layer_stat=True)
    # print('Flops:  ' + flops)
    # print('Params: ' + params)


if __name__ == '__main__':
    calc_flops_params()
