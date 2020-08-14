import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

def get_inplanes():
    return [64, 128, 256, 512]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, csn_type='3d'):
        super(BasicBlock, self).__init__()
        self.conv1 = self.add_conv(inplanes, planes, stride, csn_type=csn_type)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.add_conv(planes, planes, csn_type=csn_type)
        self.bn2 = nn.BatchNorm3d(planes)
        self.short_cut = None

        if inplanes != planes:
            self.short_cut = nn.Sequential(
                nn.Conv3d(
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm3d(planes))

    def add_conv(self, inplanes, planes, k_size=3, stride=(1, 1, 1), padding=(1, 1, 1), csn_type='3d'):

        if csn_type == '3d':
            return nn.Conv3d(inplanes, planes, k_size, stride=stride, padding=padding, bias=False) # 3d类型卷积通道不分组
        elif csn_type == 'ir':
            return nn.Conv3d(inplanes, planes, k_size, stride=stride, padding=padding, groups=inplanes, bias=False) # ir-csn卷积通道要分组，组数和输入通道数一样
        elif csn_type == 'ip':
            return nn.Sequential(
                nn.Conv3d(inplanes, 
                planes, 
                kernel_size=(1, 1, 1), 
                stride=(1, 1, 1), 
                padding=(0, 0, 0), 
                bias=False),
                nn.BatchNorm3d(planes), 
                nn.Conv3d(planes, 
                planes, 
                k_size,
                stride, 
                padding, 
                bias=False, 
                groups=planes)) # 先经过1*1*1的pointwise，再经过3*3*3的depthwise

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.short_cut is not None:
            residual = self.short_cut(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, csn_type='ip'):
        super(Bottleneck, self).__init__()

        self.conv1 = self.add_conv(inplanes, planes, k_size=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = self.add_conv(planes, planes, k_size=3, 
                                stride=stride, padding=(1, 1, 1), csn_type=csn_type)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = self.add_conv(planes, planes * 4, k_size=1)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.short_cut = None

        if inplanes != planes * 4:
            self.short_cut = nn.Sequential(
                nn.Conv3d(
                    inplanes,
                    planes * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm3d(planes * 4))

    def add_conv(self, inplanes, planes, k_size, stride=(1, 1, 1), padding=(0, 0, 0), csn_type='3d'):

        if csn_type == '3d':
            return nn.Conv3d(inplanes, planes, k_size, stride=stride, padding=padding, bias=False)
        elif csn_type == 'ir':
            return nn.Conv3d(inplanes, planes, k_size, stride=stride, padding=padding, groups=inplanes, bias=False)
        elif csn_type == 'ip':
            return nn.Sequential(
                nn.Conv3d(inplanes, 
                planes, 
                kernel_size=(1, 1, 1), 
                stride=(1, 1, 1), 
                padding=(0, 0, 0), 
                bias=False),
                nn.BatchNorm3d(planes), 
                nn.Conv3d(planes, 
                planes, 
                k_size,
                stride, 
                padding, 
                bias=False, 
                groups=planes))        

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.short_cut is not None:
            residual = self.short_cut(x)

        out += residual
        out = self.relu(out)

        return out


class CSN(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_classes=400, 
                 csn_type='3d'):
        self.inplanes = 64
        self.csn_type = csn_type
        super(CSN, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride, csn_type=self.csn_type))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, csn_type=self.csn_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.maxpool(x)

        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = CSN(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = CSN(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = CSN(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = CSN(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = CSN(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = CSN(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = CSN(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

if __name__ == "__main__":
    CSN = generate_model(model_depth=152, csn_type='ip')
    print(CSN)
    CSN.load_state_dict(torch.load('data/models/model_ip.pkl'))
    print('load complete')
    # CSN.load_state_dict(torch.load('../pretrained/ipCSN_152_ft_kinetics_from_ig65m_trans.pkl'))
    # torch.save(CSN.state_dict(), '../../model_ip.pkl')
    # torch.save({'epoch':10,
    #             'state_dict':CSN.module.state_dict()}, 'check.pth.tar')
    # input = torch.randn((1, 3, 8, 224, 224))
    # out = CSN(input)
    # print(out.shape)
