# -*- coding: utf-8 -*-
# !/usr/bin/env python
import numpy as np
from func.ZDT1 import ZDT1
import matplotlib.pyplot as plt
import zdt1_val
import random
def combine(pop,pop_offspring):
    pop_offspring=np.array(pop_offspring)

    pop_combine= np.vstack((pop, pop_offspring))

    return pop_combine
def cross_mutation(pop_parant, yN, alfa, belta):
    n = len(pop_parant)
    pop_offspring = []

    while len(pop_offspring) <= n:
        a, b = random.randint(0, n - 1), random.randint(0, n - 1)
        p1, p2 = pop_parant[a], pop_parant[b]
        if (p1 == p2).all():
            pass
        # 交叉
        elif random.random() < alfa:
            pop_bq = []
            for i in range(int(yN)):
                u = random.random()
                if u < 0.5:
                    bq = (2 * u) ** (1 / 3)
                else:
                    bq = (1/(2 * (1 - u))) ** (1 / 3)
                pop_bq.append(bq)
            pop_bq = np.array(pop_bq)
            c1 = 0.5 * ((1 + pop_bq) * p1 + (1 - pop_bq) * p2)
            c2 = 0.5 * ((1 + pop_bq) * p2 + (1 - pop_bq) * p1)
            c1[c1 > 1] = 1
            c2[c2 > 1] = 1
            c1[c1 < 0] = 0
            c2[c2 < 0] = 0
            pop_offspring.append(c1)
            pop_offspring.append(c2)
        # 变异
        elif random.random() > 1 - belta:
            c1 = p1[:]
            c2 = p2[:]
            for j in range(yN):
                u = np.random.rand()
                if u < 0.5:
                    bq = (2 * u) ** (1 / 6) - 1
                else:
                    bq = 1 - (2 * (1 - u)) ** (1 / 6)
                if c1[j] + bq < 0:
                    c1[j] = 0
                elif c1[j] + bq > 1:
                    c1[j] = 1
                else:
                    c1[j] = bq + c1[j]

            for j in range(yN):
                u = np.random.rand()
                if u < 0.5:
                    bq = (2 * u) ** (1 / 2) - 1
                else:
                    bq = 1 - (2 * (1 - u)) ** (1 / 2)
                if c2[j] + bq < 0:
                    c2[j] = 0
                elif c2[j] + bq > 1:
                    c2[j] = 1
                else:
                    c2[j] = bq + c2[j]
            pop_offspring.append(c1)
            pop_offspring.append(c2)

    return pop_offspring
from collections import OrderedDict


def crowding_distance_sort(pop_layer_index_f1_f2):
    pop_layer_index_distance = {}
    for k, v in pop_layer_index_f1_f2.items():
        pop_index_distance = OrderedDict()
        f1 = sorted(v, key=lambda x: x[1][0])
        f2 = sorted(v, key=lambda x: x[1][1])
        f1_max = f1[-1][1][0]
        f1_min = f1[0][1][0]
        f2_max = f2[-1][1][1]
        f2_min = f2[0][1][1]
        f1_l = len(f1)

        if f1_l == 1:
            pop_index_distance[f1[0][0]] = (float("inf"))
        elif f1_l == 2:
            pop_index_distance[f1[0][0]] = (float("inf"))
            pop_index_distance[f1[1][0]] = (float("inf"))
        else:
            pop_index_distance[f1[0][0]] = (float("inf"))
            pop_index_distance[f1[-1][0]] = (float("inf"))
            for j in range(f1_l - 2):
                if f1_max - f1_min == 0:
                    f1_d = 0
                else:
                    f1_d = (f1[j + 2][1][0] - f1[j][1][0]) / (f1_max - f1_min)
                pop_index_distance[f1[j + 1][0]] = f1_d

            for i in range(f1_l - 2):
                if f2_max - f2_min == 0:
                    f2_d = 0
                else:
                    f2_d = (f2[i + 2][1][1] - f2[i][1][1]) / (f2_max - f2_min)
                pop_index_distance[f1[i + 1][0]] += f2_d

        pop_layer_index_distance[k] = pop_index_distance

    for k in list(pop_layer_index_distance.keys()):
        pop_layer_index_distance[k] = sorted(pop_layer_index_distance[k].items(), key=lambda x: x[1], reverse=True)

    pop_index_layer_distance_sorted = []
    for k, v in pop_layer_index_distance.items():
        for i in v:
            index_distance_layer = [0, 0, 0]
            index_distance_layer[0] = i[0]
            index_distance_layer[1] = k
            index_distance_layer[2] = i[1]
            pop_index_layer_distance_sorted.append(index_distance_layer)

    return pop_index_layer_distance_sorted
def elitism(pop_combine_index_layer_distance_sorted,pop, xN):
    pop_combine_index_layer_distance_sorted = pop_combine_index_layer_distance_sorted[0:xN]
    # print(pop_combine_index_layer_distance_sorted)
    pop_new=[]
    pop_new_index_layer_distance=[]
    for i in range(xN):
        index=pop_combine_index_layer_distance_sorted[i][0]
        layer=pop_combine_index_layer_distance_sorted[i][1]
        distance = pop_combine_index_layer_distance_sorted[i][2]
        pop_new.append(pop[index])

        pop_new_index_layer_distance.append((i,layer,distance))

    return pop_new_index_layer_distance,pop_new
def non_domination_sort(pop):
    pop_index_f1_f2 = {}
    n = len(pop)
    for i in range(n):
        val = ZDT1(pop[i])
        pop_index_f1_f2[i] = val

    pop_index_layer_f1_f2 = control_relationship(pop_index_f1_f2)

    pop_layer_index_f1_f2 = creat_layer(pop_index_layer_f1_f2, pop_index_f1_f2)

    return pop_layer_index_f1_f2


def control_relationship(FuncValueList):
    control_1_dict = {}
    # k0就是pop的索引
    for k0, v0 in FuncValueList.items():
        control_1_dict[k0] = {"支配": [], "被支配": [], "相等": []}
        for k1, v1 in FuncValueList.items():
            if k0 == k1:
                pass
            elif v0[0] < v1[0]:
                control_1_dict[k0]["支配"].append(k1)
            elif v0[0] > v1[0]:
                control_1_dict[k0]["被支配"].append(k1)
            else:
                control_1_dict[k0]["相等"].append(k1)

    control_2_dict = {}

    for k0, v0 in FuncValueList.items():
        control_2_dict[k0] = {"支配": [], "被支配": [], "相等": []}
        for k1, v1 in FuncValueList.items():
            if v0[1] < v1[1]:
                control_2_dict[k0]["支配"].append(k1)
            elif v0[1] > v1[1]:
                control_2_dict[k0]["被支配"].append(k1)
            else:
                if k0 != k1:
                    control_2_dict[k0]["相等"].append(k1)

    control_list = {}
    for index in FuncValueList.keys():
        control_list[index] = {"被支配集合": [], "支配集合": []}
        # set.intersection 交集
        # 强支配关系
        control_list[index]["支配集合"] = set.intersection(set(control_1_dict[index]["支配"]),
                                                       set(control_2_dict[index]["支配"]))
        control_list[index]["被支配集合"] = set.intersection(set(control_1_dict[index]["被支配"]),
                                                        set(control_2_dict[index]["被支配"]))
        # set.union 并集
        # 弱支配关系

    for k, v in control_list.items():
        v["被支配集合"] = len(v["被支配集合"])
    return control_list
def creat_layer(control_list, pop_index_f1_f2):
    layer = 0
    control_layer = {}
    while control_list:
        # 某一层的元素
        layer_val = []
        for k in list(control_list.keys()):
            if control_list[k]['被支配集合'] == 0:
                layer_val.append((k, pop_index_f1_f2[k]))
                control_list.pop(k)
            else:
                control_list[k]['被支配集合'] -= 1
        control_layer[layer] = layer_val
        if layer_val:
            layer += 1
    return control_layer
def tournament_selection(pop_combine_index_layer_distance_sorted, pop):
    # 二元竞赛
    select_pop = []
    n = len(pop_combine_index_layer_distance_sorted)
    while len(select_pop) < n:
        a, b = random.randint(0, n - 1), random.randint(0, n - 1)
        if pop_combine_index_layer_distance_sorted[a][1] < pop_combine_index_layer_distance_sorted[b][1]:
            select_pop.append(pop[pop_combine_index_layer_distance_sorted[a][0]])
        elif pop_combine_index_layer_distance_sorted[a][1] == pop_combine_index_layer_distance_sorted[b][1]:
            if pop_combine_index_layer_distance_sorted[a][2] > pop_combine_index_layer_distance_sorted[b][2]:
                select_pop.append(pop[pop_combine_index_layer_distance_sorted[a][0]])
        else:
            select_pop.append(pop[pop_combine_index_layer_distance_sorted[b][0]])

    return select_pop
val = np.array(zdt1_val.data) #理论值
gen = 300  # 迭代次数
xN = 100  # 种群数量
yN = 30  # 变量个数
alfa = 0.9
belta = 0.1
pop = np.random.rand(xN * yN).reshape(xN, yN) * 1.0  # 初始化种群
pop_index_layer_f1_f2 = non_domination_sort(pop)#非支配排序
pop_index_layer_distance_sorted = crowding_distance_sort(pop_index_layer_f1_f2)#首先进行拥挤度计算
#plt.ion()
plt.figure(figsize=(7, 4))
for i in range(gen):
    # 二进制锦标赛
    pop_parant = tournament_selection(pop_index_layer_distance_sorted, pop)#根据拥挤度来进行选择
    # tour_time=time.time()
    # print("tour_time",tour_time-start)
    # 交叉变异，生成子代
    pop_offspring = cross_mutation(pop_parant, yN, alfa, belta)
    # 子代和上一代合并
    pop = combine(pop, pop_offspring)
    # 非支配排序
    pop_combine_index_layer_f1_f2 = non_domination_sort(pop)
    # 拥挤度排序
    pop_combine_index_layer_distance_sorted = crowding_distance_sort(pop_combine_index_layer_f1_f2)
    # 精英保留产生下一代种群
    pop_index_layer_distance_sorted, pop = elitism(pop_combine_index_layer_distance_sorted, pop, xN)

    # 最后一代，输出最前沿结果集
    #if i %50==0:
    pop_index_f1_f2 = []
    print("第{}代".format(i))
    for i in pop_index_layer_distance_sorted:
        index = i[0]
        if i[1] == 0:
            pop_index_f1_f2.append(ZDT1(pop[index]))
    pop_index_f1_f2 = np.array(pop_index_f1_f2)
    x = pop_index_f1_f2[:, 0]
    y = pop_index_f1_f2[:, 1]
    x1 = val[:, 0]
    y1 = val[:, 1]
    plt.plot(x, y, 'ro')
    plt.plot(x1, y1, 'ro', c='b')
    plt.grid(True)
    plt.axis('tight')
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.pause(0.01)
    plt.clf()

    #plt.ioff()
plt.show()
