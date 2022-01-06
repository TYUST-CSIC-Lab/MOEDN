# -*- coding: utf-8 -*-
import random
import csv
"""
    @date: 2019.11.13
    @function: hyparameter for training & inference.
"""
# r = 20
# for rr in range(r):
#     j = random.randint(2, 5)
#     a = random.randint(2,20)
#     b = random.randint(1,30)
#     # c = random.uniform(0.1, 0.2)
#     c = random.random()
#     d = (random.random())/100
#     # list = [20, 50, 100]
#     # i = random.randint(0, 2)
#     # e = list[i]
#     e = random.randint(20, 100)
#     # list = [0, 0.1, 0.2, 0.3]
#     # ii = random.randint(0, 3)
#     # f = list[ii]

FLAGS = {"start_epoch": 0,
         "target_epoch": 10,
         "device": "cuda",
         "mask_path": "utils/uv_data/uv_weight_mask_gdh.png",
         "block_size": 36,
         "alpha": 48,
         "dist_prob": 0.492,
         "nr_steps": 5e3,
         "lr": 0.00005,
         "batch_size": 64,
         "save_interval": 5,
         "normalize_mean": [0.485, 0.456, 0.406],
         "normalize_std": [0.229, 0.224, 0.225],
         "images": "H:\Code\python\PRNet_PyTorch-master/results",
         "gauss_kernel": "original",
         "summary_path": "H:\Code\python\PRNet_PyTorch-master\prnet_runs",
         "summary_step": 0,
         "resume": True}
    # parameters = []
    # parameters.append(a)
    # parameters.append(b)
    # parameters.append(c)
    # parameters.append(d)
    # # parameters.append(c)
    # # parameters.append(f)
    # with open('H:\Code\python\PRNet_PyTorch-master\paramater.csv', 'a+', newline='', encoding='utf-8') as f:
    #     f_csv = csv.writer(f)
    #     f_csv.writerow(parameters)