# -*- encoding = utf-8 -*-
"""
@description: 构造数据集
@date: 2023/10/17
@File : dataset.py
@Software : PyCharm
"""
import os
import random
from dhg.structure.graphs import Graph
import dhg
import numpy as np
import torch
import pandas as pd
import pickle

"""
思路：
1、首先，以一个class为中心，和它相关的move method的class都找出来，组成一个graph
2、然后，需要获取跨class的依赖结构，由此组成低阶图
3、move method之前，每一个class组成高阶图是负样本；move method之后组成高阶图是正样本，这样正负样本是平衡的
4、使用pre-trained model获取图中每个节点的特征(field是否考虑，考虑可能会变麻烦)
5、将图数据转化为DHG框架的格式
"""

train_set = []
val_set = []
test_set = []
ant = []
derby = []
drjava = []
jfreechart = []
jgroups = []
jhotdraw = []
jtopen = []
junit = []
lucene = []
mvnforum = []
tapestry = []



def create_digraph(path):
    edges = []
    v = 0
    with open(path + '\\graph_node.csv', "r", encoding="utf-8") as gr:
        # 查看graph_node.csv一共有几行数据
        lines = gr.readlines()
        v += len(lines)
    if not os.path.exists(path + '\\relation.csv'):
        return dhg.Graph(v, edges)
    with open(path + '\\relation.csv', "r", encoding="utf-8") as pr:
        for line in pr:
            line = line.strip('\n').split(',')
            edges.append([int(line[0]), int(line[1])])
    # 将edge转换为类型：Union[list[int], list[list[int]], None]

    return dhg.Graph(v, edges)


def create_hypergraph(path, tag):
    source_id = 0
    with open(path + "\\extract_range.csv", 'r', encoding="utf-8") as er:
        line = er.readline()
        line = line.strip('\n').split(',')
        source_class = line[0]
    with open(path + "\\target_range.csv", 'r', encoding="utf-8") as tr:
        line = tr.readline()
        line = line.strip('\n').split(',')
        target_class = line[0]
    edges = []
    with open(path + "\\graph_node.csv", 'r', encoding="utf-8") as gr:
        for line in gr:
            line = line.strip('\n').split(',')
            no_tag = line[8]
            if no_tag == "1":
                source_id = int(line[0])
                break
    edges.append(source_id)
    if tag == "1" or tag == "0":
        v = 0
        with open(path + "\\graph_node.csv", 'r', encoding="utf-8") as gr:
            for line in gr:
                v += 1
                line = line.strip('\n').split(',')
                no_class = line[2].split('-')[0]
                no_tag = line[8]
                if no_tag == "1":
                    continue
                # 1构造正样本
                if tag == "1" and no_class == target_class:
                    edges.append(int(line[0]))
                elif tag == "0" and no_class == source_class:
                    edges.append(int(line[0]))
        flag = 0 if len(edges) == 1 else 1
        return 1, dhg.Hypergraph(v, [edges])

    hgs = []
    v = 0
    with open(path + "\\graph_node.csv", 'r', encoding="utf-8") as gr:
        v += len(gr.readlines())
    with open(path + "\\graph_node.csv", 'r', encoding="utf-8") as gr:
        edges = [source_id]
        pre_class = "-1"
        for line in gr:
            line = line.strip('\n').split(',')
            no_class = line[2].split('-')[0]
            no_tag = line[8]
            if no_tag == "1" or no_class == target_class or no_class == source_class:
                continue
            if pre_class == "-1":
                pre_class = no_class
                edges.append(int(line[0]))
            if no_class != pre_class:
                hg = dhg.Hypergraph(v, [edges])
                hgs.append(hg)
                edges.clear()
                edges.append(source_id)
                pre_class = no_class
        if len(edges) != 1:
            hg = dhg.Hypergraph(v, [edges])
            hgs.append(hg)
    return 0, hgs


def l1_mormalize(features):
    return torch.nn.functional.normalize(features, p=1, dim=1)


# codebert codegpt codet5 codet5plus codetrans cotext graphcodebert plbart
def get_features(path, name):
    features = []
    with open(path + "\\embedding\\" + name + ".csv", 'r', encoding="utf-8") as gr:
        for line in gr:
            line = [float(x) for x in line.strip('\n').split(',')]
            features.append(line)
    features_tensor = torch.tensor(features)
    # if name != "codet5plus":
    #
    features_tensor = l1_mormalize(features_tensor)
    return features_tensor


def solve(path, tag, emb_type):
    global train_set
    global val_set
    global test_set
    g = create_digraph(path)
    flag1, hg = create_hypergraph(path, '1')
    features = get_features(path, emb_type)
    flag2, hg1 = create_hypergraph(path, '0')
    # print(path, hg.e.count(0), hg1.e.count(0))
    # g1 = Graph.from_hypergraph_hypergcn(
    #     hg, features, False  # , device=X.device
    # )
    # g1 = Graph.from_hypergraph_hypergcn(
    #     hg1, features, False  # , device=X.device
    # )
    # 构造随机负样本
    # hgs = create_hypergraph(path, '3')
    # for hg_new in hgs:
    #     if tag == 1:
    #         train_set.append([g, hg_new, features, torch.tensor(0)])
    #     elif tag == 2:
    #         val_set.append([g, hg_new, features, torch.tensor(0)])
    #     elif tag == 3:
    #         test_set.append([g, hg_new, features, torch.tensor(0)])
    if tag == 1:
        if flag1:
            train_set.append([g, hg, features, torch.tensor(1)])
        if flag2:
            train_set.append([g, hg1, features, torch.tensor(0)])

    elif tag == 2:
        if flag1:
            val_set.append([g, hg, features, torch.tensor(1)])
        if flag2:
            val_set.append([g, hg1, features, torch.tensor(0)])
    elif tag == 3:
        if flag1:
            test_set.append([g, hg, features, torch.tensor(1)])
        if flag2:
            test_set.append([g, hg1, features, torch.tensor(0)])
    elif tag == 4:
        if flag1:
            ant.append([g, hg, features, torch.tensor(1)])
        if flag2:
            ant.append([g, hg1, features, torch.tensor(0)])
    elif tag == 5:
        if flag1:
            derby.append([g, hg, features, torch.tensor(1)])
        if flag2:
            derby.append([g, hg1, features, torch.tensor(0)])
    elif tag == 6:
        if flag1:
            drjava.append([g, hg, features, torch.tensor(1)])
        if flag2:
            drjava.append([g, hg1, features, torch.tensor(0)])
    elif tag == 7:
        if flag1:
            jfreechart.append([g, hg, features, torch.tensor(1)])
        if flag2:
            jfreechart.append([g, hg1, features, torch.tensor(0)])
    elif tag == 8:
        if flag1:
            jgroups.append([g, hg, features, torch.tensor(1)])
        if flag2:
            jgroups.append([g, hg1, features, torch.tensor(0)])
    elif tag == 9:
        if flag1:
            jhotdraw.append([g, hg, features, torch.tensor(1)])
        if flag2:
            jhotdraw.append([g, hg1, features, torch.tensor(0)])
    elif tag == 10:
        if flag1:
            jtopen.append([g, hg, features, torch.tensor(1)])
        if flag2:
            jtopen.append([g, hg1, features, torch.tensor(0)])
    elif tag == 11:
        if flag1:
            junit.append([g, hg, features, torch.tensor(1)])
        if flag2:
            junit.append([g, hg1, features, torch.tensor(0)])
    elif tag == 12:
        if flag1:
            lucene.append([g, hg, features, torch.tensor(1)])
        if flag2:
            lucene.append([g, hg1, features, torch.tensor(0)])
    elif tag == 13:
        if flag1:
            mvnforum.append([g, hg, features, torch.tensor(1)])
        if flag2:
            mvnforum.append([g, hg1, features, torch.tensor(0)])
    elif tag == 14:
        if flag1:
            tapestry.append([g, hg, features, torch.tensor(1)])
        if flag2:
            tapestry.append([g, hg1, features, torch.tensor(0)])

def scan_dir(path, tag, emb_type):
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isdir(file_path):
            if file_name.startswith('1') or file_name.startswith('2'):
                solve(file_path, tag, emb_type)
            else:
                scan_dir(file_path, tag, emb_type)


def get_dataset(emb):
    global train_set
    global val_set
    global test_set
    train_set.clear()
    val_set.clear()
    test_set.clear()

    print("prepare dataset")
    scan_dir('E:\\HMove\\dataset\\train', 1, emb)
    # scan_dir('E:\\HMove\\dataset\\train', 1, emb)
    scan_dir('E:\\HMove\\dataset\\val', 2, emb)
    scan_dir('E:\\HMove\\dataset\\test', 3, emb)
    scan_dir('E:\\HMove\\dataset\\test\\ant', 4, emb)
    scan_dir('E:\\HMove\\dataset\\test\\derby', 5, emb)
    scan_dir('E:\\HMove\\dataset\\test\\drjava', 6, emb)
    scan_dir('E:\\HMove\\dataset\\test\\jfreechart', 7, emb)
    scan_dir('E:\\HMove\\dataset\\test\\jgroups', 8, emb)
    scan_dir('E:\\HMove\\dataset\\test\\jhotdraw', 9, emb)
    scan_dir('E:\\HMove\\dataset\\test\\jtopen', 10, emb)
    scan_dir('E:\\HMove\\dataset\\test\\junit', 11, emb)
    scan_dir('E:\\HMove\\dataset\\test\\lucene', 12, emb)
    scan_dir('E:\\HMove\\dataset\\test\\mvnforum', 13, emb)
    scan_dir('E:\\HMove\\dataset\\test\\tapestry', 14, emb)
    print(f'train_set: {len(train_set)}, val_set: {len(val_set)}, test_set: {len(test_set)}')
    return train_set, val_set, test_set, ant, derby, drjava, jfreechart, jgroups, jhotdraw, jtopen, junit, lucene, mvnforum, tapestry
