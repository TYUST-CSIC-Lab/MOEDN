# -*- coding: utf-8 -*-
"""
    @author: samuel ko
    @date: 2019.07.18
    @readme: The implementation of PRNet Network

    @notice: PyTorch only support odd convolution to keep half downsample.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import *
from model.disout import Disout,LinearScheduler
from config.config import FLAGS
import numpy as np
import math


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding='same'):
    """3x3 convolution with padding"""
    if padding == 'same':
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False, dilation=dilation)


class BasicBlock_disout(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1,downsample=None,
                 kernel_size=3, dist_prob=0.1, block_size=2, alpha=3, nr_steps=5e3,
                 norm_layer=None):
        super(BasicBlock_disout, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.shortcut_conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.disout0=LinearScheduler(Disout(dist_prob=dist_prob,block_size=block_size,alpha=alpha),
                                     start_value=0.,stop_value=dist_prob,nr_steps=nr_steps)
        self.conv1 = nn.Conv2d(inplanes, planes // 2, kernel_size=1, stride=1, padding=0)
        self.disout1=LinearScheduler(Disout(dist_prob=dist_prob,block_size=block_size,alpha=alpha),
                                     start_value=0.,stop_value=dist_prob,nr_steps=nr_steps)
        #self.normalizer_fn1 = norm_layer(planes//2)
        self.conv2 = nn.Conv2d(planes // 2, planes // 2, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2)
        self.disout2=LinearScheduler(Disout(dist_prob=dist_prob,block_size=block_size,alpha=alpha),
                                     start_value=0.,stop_value=dist_prob,nr_steps=nr_steps)
        #self.normalizer_fn2 = norm_layer(planes//2)
        self.conv3 = nn.Conv2d(planes // 2, planes, kernel_size=1, stride=1, padding=0)
        self.disout3=LinearScheduler(Disout(dist_prob=dist_prob,block_size=block_size,alpha=alpha),
                                     start_value=0.,stop_value=dist_prob,nr_steps=nr_steps)
        self.normalizer_fn3 = norm_layer(planes)
        self.activation_fn = nn.ReLU(inplace=True)
        self.downsample=downsample
        self.stride = stride
        self.disout4=LinearScheduler(Disout(dist_prob=dist_prob,block_size=block_size,alpha=alpha),
                                     start_value=0.,stop_value=dist_prob,nr_steps=nr_steps)
        self.out_planes = planes

    def forward(self, x):
        # shortcut = x
        # (_, _, _, x_planes) = x.size()
        #
        # if self.stride != 1 or x_planes != self.out_planes:
        #     shortcut = self.shortcut_conv(x)
        #
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        #
        # x += shortcut
        # x = self.normalizer_fn(x)
        # x = self.activation_fn(x)

        # return x
        shortcut = x
        (_, _, _, x_planes) = x.size()

        if self.stride != 1 or x_planes != self.out_planes:
            shortcut = self.shortcut_conv(x)
            shortcut=self.disout0(shortcut)
        x = self.conv1(x)
        #x = self.normalizer_fn1(x)
        #x = self.activation_fn(x)
        x = self.disout1(x)

        x = self.conv2(x)
       # x = self.normalizer_fn2(x)
        #x = self.activation_fn(x)
        x = self.disout2(x)

        x = self.conv3(x)
        #x = self.normalizer_fn3(x)
        x = self.disout3(x)

        if self.downsample is not None:
            shortcut = self.downsample(x)
        shortcut = self.disout4(shortcut)
        x += shortcut
        x = self.normalizer_fn3(x)
        x = self.activation_fn(x)

        return x

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 kernel_size=3,
                 norm_layer=None):
        super(ResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.shortcut_conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.conv1 = nn.Conv2d(inplanes, planes // 2, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(planes // 2, planes // 2, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.conv3 = nn.Conv2d(planes // 2, planes, kernel_size=1, stride=1, padding=0)

        self.normalizer_fn = norm_layer(planes)
        self.activation_fn = nn.ReLU(inplace=True)

        self.stride = stride
        self.out_planes = planes

    def forward(self, x):
        shortcut = x
        (_, _, _, x_planes) = x.size()

        if self.stride != 1 or x_planes != self.out_planes:
            shortcut = self.shortcut_conv(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x += shortcut
        x = self.normalizer_fn(x)
        x = self.activation_fn(x)

        return x


class ResFCN256(nn.Module):
    def __init__(self, resolution_input=256, resolution_output=256, channel=3, size=16):
        super().__init__()
        self.input_resolution = resolution_input
        self.output_resolution = resolution_output
        self.channel = channel
        self.size = size

        # Encoder
        self.block0 = conv3x3(in_planes=3, out_planes=self.size, padding='same')
        self.block1 = ResBlock(inplanes=self.size, planes=self.size * 2, stride=2)
        self.block2 = ResBlock(inplanes=self.size * 2, planes=self.size * 2, stride=1)
        self.block3 = ResBlock(inplanes=self.size * 2, planes=self.size * 4, stride=2)
        self.block4 = ResBlock(inplanes=self.size * 4, planes=self.size * 4, stride=1)
        self.block5 = ResBlock(inplanes=self.size * 4, planes=self.size * 8, stride=2)
        self.block6 = ResBlock(inplanes=self.size * 8, planes=self.size * 8, stride=1)
        self.block7 = ResBlock(inplanes=self.size * 8, planes=self.size * 16, stride=2)
        self.block8 = ResBlock(inplanes=self.size * 16, planes=self.size * 16, stride=1)
        self.block9 = ResBlock(inplanes=self.size * 16, planes=self.size * 32, stride=2)
        self.block10 = BasicBlock_disout(inplanes=self.size * 32, planes=self.size * 32, stride=1,dist_prob=0.09, block_size=6, alpha=5, nr_steps=5e3)

       # Decoder
        self.upsample0 = nn.ConvTranspose2d(self.size * 32, self.size * 32, kernel_size=3, stride=1,
                                            padding=1)  # keep shape invariant.
        self.upsample1 = nn.ConvTranspose2d(self.size * 32, self.size * 16, kernel_size=4, stride=2,
                                            padding=1)  # half downsample.
        self.upsample2 = nn.ConvTranspose2d(self.size * 16, self.size * 16, kernel_size=3, stride=1,
                                            padding=1)  # keep shape invariant.
        self.upsample3 = nn.ConvTranspose2d(self.size * 16, self.size * 16, kernel_size=3, stride=1,
                                            padding=1)  # keep shape invariant.

        self.upsample4 = nn.ConvTranspose2d(self.size * 16, self.size * 8, kernel_size=4, stride=2,
                                            padding=1)  # half downsample.
        self.upsample5 = nn.ConvTranspose2d(self.size * 8, self.size * 8, kernel_size=3, stride=1,
                                            padding=1)  # keep shape invariant.
        self.upsample6 = nn.ConvTranspose2d(self.size * 8, self.size * 8, kernel_size=3, stride=1,
                                            padding=1)  # keep shape invariant.

        self.upsample7 = nn.ConvTranspose2d(self.size * 8, self.size * 4, kernel_size=4, stride=2,
                                            padding=1)  # half downsample.
        self.upsample8 = nn.ConvTranspose2d(self.size * 4, self.size * 4, kernel_size=3, stride=1,
                                            padding=1)  # keep shape invariant.
        self.upsample9 = nn.ConvTranspose2d(self.size * 4, self.size * 4, kernel_size=3, stride=1,
                                            padding=1)  # keep shape invariant.

        self.upsample10 = nn.ConvTranspose2d(self.size * 4, self.size * 2, kernel_size=4, stride=2,
                                             padding=1)  # half downsample.
        self.upsample11 = nn.ConvTranspose2d(self.size * 2, self.size * 2, kernel_size=3, stride=1,
                                             padding=1)  # keep shape invariant.

        self.upsample12 = nn.ConvTranspose2d(self.size * 2, self.size, kernel_size=4, stride=2,
                                             padding=1)  # half downsample.
        self.upsample13 = nn.ConvTranspose2d(self.size, self.size, kernel_size=3, stride=1,
                                             padding=1)  # keep shape invariant.

        self.upsample14 = nn.ConvTranspose2d(self.size, self.channel, kernel_size=3, stride=1,
                                             padding=1)  # keep shape invariant.
        self.upsample15 = nn.ConvTranspose2d(self.channel, self.channel, kernel_size=3, stride=1,
                                             padding=1)  # keep shape invariant.
        self.upsample16 = nn.ConvTranspose2d(self.channel, self.channel, kernel_size=3, stride=1,
                                             padding=1)  # keep shape invariant.

        # ACT
        self.sigmoid = nn.Sigmoid()
        # for name,m in self.named_modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m,nn.BatchNorm2d) and 'bn3'in name:
        #         m.weight.data.fill_(0)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    # def _init_weight(self):
    #     # init layer parameters
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         # elif isinstance(m, nn.Linear):
    #         #     m.bias.data.zero_()
    def forward(self, x):
        if self.training:
            modulelist=list(self.modules())
            num_module=len(modulelist)
            dploc=[]
            convloc=[]
            for idb in range(num_module):
                if isinstance(modulelist[idb],Disout):
                    dploc.append(idb)
                    for iconv in range(idb,num_module):
                        if isinstance (modulelist[iconv],nn.Conv2d):
                            convloc.append(iconv)
                            break
            dploc=dploc[:len(convloc)]
            assert len(dploc)==len(convloc)
            for imodu in range(len(dploc)):
                modulelist[dploc[imodu]].weight_behind=modulelist[convloc[imodu]].weight.data

            for module in self.modules():
                if isinstance(module,LinearScheduler):
                    module.step()
        se = self.block0(x)  # 256 x 256 x 16
        se = self.block1(se)  # 128 x 128 x 32
        se = self.block2(se)  # 128 x 128 x 32
        se = self.block3(se)  # 64 x 64 x 64
        se = self.block4(se)  # 64 x 64 x 64
        se = self.block5(se)  # 32 x 32 x 128
        se = self.block6(se)  # 32 x 32 x 128
        se = self.block7(se)  # 16 x 16 x 256
        se = self.block8(se)  # 16 x 16 x 256
        se = self.block9(se)  # 8 x 8 x 512
        se = self.block10(se)  # 8 x 8 x 512

        pd = self.upsample0(se)  # 8 x 8 x 512
        pd = self.upsample1(pd)  # 16 x 16 x 256
        pd = self.upsample2(pd)  # 16 x 16 x 256
        pd = self.upsample3(pd)  # 16 x 16 x 256
        pd = self.upsample4(pd)  # 32 x 32 x 128
        pd = self.upsample5(pd)  # 32 x 32 x 128
        pd = self.upsample6(pd)  # 32 x 32 x 128
        pd = self.upsample7(pd)  # 64 x 64 x 64
        pd = self.upsample8(pd)  # 64 x 64 x 64
        pd = self.upsample9(pd)  # 64 x 64 x 64

        pd = self.upsample10(pd)  # 128 x 128 x 32
        pd = self.upsample11(pd)  # 128 x 128 x 32
        pd = self.upsample12(pd)  # 256 x 256 x 16
        pd = self.upsample13(pd)  # 256 x 256 x 16
        pd = self.upsample14(pd)  # 256 x 256 x 3
        pd = self.upsample15(pd)  # 256 x 256 x 3
        pos = self.upsample16(pd)  # 256 x 256 x 3

        pos = self.sigmoid(pos)
        return pos
