# -*- coding: utf-8 -*-

"""
    @date: 2019.07.18
    @author: samuel ko
    @func: PRNet Training Part.
"""
import os
from collections import Set

import cv2

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import *
from model.disout import Disout, LinearScheduler
import numpy as np
import math
import csv
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from PIL import Image
import torch
import torch.optim
#from model.resfcn256 import ResFCN256

from tools.WLP300dataset import PRNetDataset, ToTensor, ToNormalize
from tools.prnet_loss import WeightMaskLoss, INFO

import os
from utils.utils import save_image, test_data_preprocess, make_all_grids, make_grid
from utils.losses import SSIM
import time
#from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils, models
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding='same'):
    """3x3 convolution with padding"""
    if padding == 'same':
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False, dilation=dilation)


class BasicBlock_disout(nn.Module):
    expansion = 1

    def __init__(self,FLAG,inplanes, planes, stride=1, downsample=None,
                 kernel_size=3,

                 norm_layer=None):
        self.dist_prob=FLAG["dist_prob"]
        self.block_size=FLAG["block_size"]
        self.alpha=FLAG["alpha"]
        self.nr_steps=FLAG["nr_steps"]
        super(BasicBlock_disout, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.shortcut_conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.disout0 = LinearScheduler(Disout(dist_prob=self.dist_prob, block_size=self.block_size, alpha=self.alpha),
                                       start_value=0., stop_value=self.dist_prob, nr_steps=self.nr_steps)
        self.conv1 = nn.Conv2d(inplanes, planes // 2, kernel_size=1, stride=1, padding=0)
        self.disout1 = LinearScheduler(Disout(dist_prob=self.dist_prob, block_size=self.block_size, alpha=self.alpha),
                                       start_value=0., stop_value=self.dist_prob, nr_steps=self.nr_steps)
        # self.normalizer_fn1 = norm_layer(planes//2)
        self.conv2 = nn.Conv2d(planes // 2, planes // 2, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2)
        self.disout2 = LinearScheduler(Disout(dist_prob=self.dist_prob, block_size=self.block_size, alpha=self.alpha),
                                       start_value=0., stop_value=self.dist_prob, nr_steps=self.nr_steps)
        # self.normalizer_fn2 = norm_layer(planes//2)
        self.conv3 = nn.Conv2d(planes // 2, planes, kernel_size=1, stride=1, padding=0)
        self.disout3 = LinearScheduler(Disout(dist_prob=self.dist_prob, block_size=self.block_size, alpha=self.alpha),
                                       start_value=0., stop_value=self.dist_prob, nr_steps=self.nr_steps)
        self.normalizer_fn3 = norm_layer(planes)
        self.activation_fn = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.disout4 = LinearScheduler(Disout(dist_prob=self.dist_prob, block_size=self.block_size, alpha=self.alpha),
                                       start_value=0., stop_value=self.dist_prob, nr_steps=self.nr_steps)
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
            shortcut = self.disout0(shortcut)
        x = self.conv1(x)
        # x = self.normalizer_fn1(x)
        # x = self.activation_fn(x)
        x = self.disout1(x)

        x = self.conv2(x)
        # x = self.normalizer_fn2(x)
        # x = self.activation_fn(x)
        x = self.disout2(x)

        x = self.conv3(x)
        # x = self.normalizer_fn3(x)
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
        self.conv2 = nn.Conv2d(planes // 2, planes // 2, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2)
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
    def __init__(self,FLAG,resolution_input=256, resolution_output=256, channel=3, size=16):
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
        self.block10 = BasicBlock_disout(FLAG,inplanes=self.size * 32, planes=self.size * 32, stride=1,
)

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

    def _init_weight(self):
        # init layer parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()

    def forward(self, x):
        if self.training:
            modulelist = list(self.modules())
            num_module = len(modulelist)
            dploc = []
            convloc = []
            for idb in range(num_module):
                if isinstance(modulelist[idb], Disout):
                    dploc.append(idb)
                    for iconv in range(idb, num_module):
                        if isinstance(modulelist[iconv], nn.Conv2d):
                            convloc.append(iconv)
                            break
            dploc = dploc[:len(convloc)]
            assert len(dploc) == len(convloc)
            for imodu in range(len(dploc)):
                modulelist[dploc[imodu]].weight_behind = modulelist[convloc[imodu]].weight.data

            for module in self.modules():
                if isinstance(module, LinearScheduler):
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


#Set random seem for reproducibility
manualSeed = 3
INFO("Random Seed", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def main(data_dir,FLAG):
    # -*- coding: utf-8 -*-
    """
        @author: samuel ko
        @date: 2019.07.18
        @readme: The implementation of PRNet Network

        @notice: PyTorch only support odd convolution to keep half downsample.
    """


    # 0) Tensoboard Writer.
    # writer = SummaryWriter(FLAG['summary_path'])
    origin_img, uv_map_gt, uv_map_predicted = None, None, None

    if not os.path.exists(FLAG['images']):
        os.mkdir(FLAG['images'])

    # 1) Create Dataset of 300_WLP & Dataloader.
    wlp300 = PRNetDataset(root_dir=data_dir,
                          transform=transforms.Compose([ToTensor(),
                                                        ToNormalize(FLAG["normalize_mean"], FLAG["normalize_std"])]))

    wlp300_dataloader = DataLoader(dataset=wlp300, batch_size=FLAG['batch_size'], shuffle=True, num_workers=8)

    # 2) Intermediate Processing.
    transform_img = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Normalize(FLAG["normalize_mean"], FLAG["normalize_std"])
    ])

    # 3) Create PRNet model.
    start_epoch, target_epoch = FLAG['start_epoch'], FLAG['target_epoch']
    model = ResFCN256(FLAG)

    # Load the pre-trained weight
    if FLAG['resume'] and os.path.exists(os.path.join(FLAG['images'], "latest.pth")):
        state = torch.load(os.path.join(FLAG['images'], "latest.pth"))
        model=torch.nn.DataParallel(model,device_ids=[0])
        device=torch.device("cuda:0")
        model.to(device)
        model.load_state_dict(state['prnet'])
        start_epoch = state['start_epoch']
        INFO("Load the pre-trained weight! Start from Epoch", start_epoch)
    else:
        start_epoch = 0
        INFO("Pre-trained weight cannot load successfully, train from scratch!")

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to("cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAG["lr"], betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma= 0.99)
    stat_loss = SSIM(mask_path=FLAG["mask_path"], gauss=FLAG["gauss_kernel"])
    loss = WeightMaskLoss(mask_path=FLAG["mask_path"])

    for ep in range(start_epoch, target_epoch):
        bar = tqdm(wlp300_dataloader)
        Loss_list, Stat_list = [], []
        for i, sample in enumerate(bar):
            uv_map, origin = sample['uv_map'].to(FLAG['device']), sample['origin'].to(FLAG['device'])
            start = time.clock()
            # Inference.
            uv_map_result = model(origin)

            # Loss & ssim stat.
            logit = loss(uv_map_result, uv_map)
            stat_logit = stat_loss(uv_map_result, uv_map)

            # Record Loss.
            Loss_list.append(logit.item())
            Stat_list.append(stat_logit.item())

            # Update.
            optimizer.zero_grad()
            logit.backward()
            optimizer.step()
            start1 = time.clock()
            end=start1-start
            bar.set_description(" {} [Loss(Paper)] {} [SSIM({})] {} [TIME[{}]]".format(ep, Loss_list[-1], FLAG["gauss_kernel"], Stat_list[-1],end))

            # Record Training information in Tensorboard.
            # if origin_img is None and uv_map_gt is None:
            #     origin_img, uv_map_gt = origin, uv_map
            # uv_map_predicted = uv_map_result
            #
            # writer.add_scalar("Original Loss", Loss_list[-1], FLAG["summary_step"])
            # writer.add_scalar("SSIM Loss", Stat_list[-1], FLAG["summary_step"])
            #
            # grid_1, grid_2, grid_3 = make_grid(origin_img, normalize=True), make_grid(uv_map_gt), make_grid(uv_map_predicted)
            #
            # writer.add_image('original', grid_1, FLAG["summary_step"])
            # writer.add_image('gt_uv_map', grid_2, FLAG["summary_step"])
            # writer.add_image('predicted_uv_map', grid_3, FLAG["summary_step"])
            # writer.add_graph(model, uv_map)

        if ep % FLAG["save_interval"] == 0:
            with torch.no_grad():
                origin = cv2.imread("test_data\obama_origin.jpg")
                gt_uv_map = np.load("test_data/test_obama.npy")
                origin, gt_uv_map = test_data_preprocess(origin), test_data_preprocess(gt_uv_map)

                origin, gt_uv_map = transform_img(origin), transform_img(gt_uv_map)

                origin_in = origin.unsqueeze_(0).cuda()
                pred_uv_map = model(origin_in).detach().cpu()

                save_image([origin.cpu(), gt_uv_map.unsqueeze_(0).cpu(), pred_uv_map],
                           os.path.join(FLAG['images'], str(ep) + '.png'), nrow=1, normalize=True)

            # Save model
            state = {
                'prnet': model.state_dict(),
                'Loss': Loss_list,
                'start_epoch': ep,
            }
            torch.save(state, os.path.join(FLAG['images'], 'latest.pth'))

            scheduler.step()
            state_list = []
            loss_list = []
            state_list.append(Stat_list[-1])
            loss_list.append(Loss_list[-1])

        # writer.close()
    return state_list, loss_list

def result(a):
    #import pandas as pd
    #data=pd.read_csv(a,header=None)
    #data.drop(data.index[0], inplace=True)
    # paramater =  np.recfromcsv(a)
    # paramater.dtype='float32'
    paramater=np.loadtxt(a,delimiter=',')
    r = len(paramater)
    fitness3 = [[0] * 2 for _ in range(30)]
    #
    #
    for rr in range(r):
        FLAG = {"start_epoch": 0,
                "target_epoch": 200,
                "device": "cuda",
                "mask_path": "H:\Code\python\PRNet_PyTorch-master/utils/uv_data/uv_weight_mask_gdh.png",
                "block_size": 36,
                "alpha": 48,
                "dist_prob": 0.492,
                "nr_steps": 5e3,
                "lr": 0.00005,
                "batch_size": 32,
                "save_interval": 5,
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
                "images": "results",
                "gauss_kernel": "original",
                "summary_path": "prnet_runs",
                "summary_step": 0,
                "resume": 0}
        start_time = time.clock()
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--train_dir", default=r'H:\Code\python\PRNet_PyTorch-master\utils/AFLW_2000',
                            help="specify input directory.")
        args = parser.parse_args()
        main(args.train_dir, FLAG)
        # print(rr)
        end_time = time.clock()

        fitness3[rr][0]=(fitness1[-1])
        fitness3[rr][1]=(fitness2[-1])
        print("Running time:", ((end_time - start_time)/r))
        print(a[-1])
    if os.path.exists('H:/Code/python/PRNet_PyTorch-master/result.csv'):
        os.remove('H:/Code/python/PRNet_PyTorch-master/result.csv')
        for rr in range(r):
            with open('H:/Code/python/PRNet_PyTorch-master/result.csv', 'a+', newline='', encoding='utf-8') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(fitness3[rr][:])
    else:
        for rr in range(r):
            with open('H:/Code/python/PRNet_PyTorch-master/result.csv', 'a+', newline='', encoding='utf-8') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(fitness3[rr][:])
    return fitness3

if __name__ == "__main__":
    # print(sys.argv)
    # if len(sys.argv) < 2:
    #     print('No file specified.')
    #     sys.exit()
    # else:
    #     f2 = result(sys.argv[1])
    #     # print(f2)
    a='H:\Code\python\PRNet_PyTorch-master\paramater.csv'
    f2=result(a)
    print(f2)