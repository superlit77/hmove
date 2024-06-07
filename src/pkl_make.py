# -*- encoding = utf-8 -*-
"""
@description: 构造数据集
@date: 2023/10/17
@File : dataset.py
@Software : PyCharm
"""
import os
import random

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
idx = 0
data = {
    "features": [],
    "g_edge_list": [],
    "hg_edge_list": [],
    "labels": [],
    "train_mask": [],
    "val_mask": [],
    "test_mask": []
}


def create_digraph(path):
    global idx
    edges = []
    v = 0
    with open(path + '\\graph_node.csv',  "r", encoding="utf-8") as gr:
        # 查看graph_node.csv一共有几行数据
        lines = gr.readlines()
        v = len(lines)
    if not os.path.exists(path + '\\relation.csv'):
        return v, [], dhg.Graph(v, edges)
    with open(path + '\\relation.csv',  "r", encoding="utf-8") as pr:
        for line in pr:
            line = line.strip('\n').split(',')
            edges.append([int(line[0])+idx, int(line[1])+idx])
    # 将edge转换为类型：Union[list[int], list[list[int]], None]

    return v, edges, dhg.Graph(v, edges)


def create_hypergraph(path, tag):
    global idx
    source_id = 0
    with open(path+"\\extract_range.csv", 'r', encoding="utf-8") as er:
        line = er.readline()
        line = line.strip('\n').split(',')
        source_class = line[0]
    with open(path+"\\target_range.csv", 'r', encoding="utf-8") as tr:
        line = tr.readline()
        line = line.strip('\n').split(',')
        target_class = line[0]
    edges = []
    v = 0
    with open(path+"\\graph_node.csv", 'r', encoding="utf-8") as gr:
        for line in gr:
            v += 1
            line = line.strip('\n').split(',')
            no_class = line[2].split('-')[0]
            no_tag = line[8]
            if no_tag == "1":
                source_id = int(line[0])
                continue
            # 1构造正样本
            if tag == "1" and no_class == target_class:
                edges.append(int(line[0])+idx)
            elif tag == "0" and no_class == source_class:
                edges.append(int(line[0])+idx)
    edges.append(source_id+idx)
    return edges, dhg.Hypergraph(v, [edges])


def l1_mormalize(features):
    return torch.nn.functional.normalize(features, p=1, dim=1)


# codebert codegpt codet5 codet5plus codetrans cotext graphcodebert plbart
def get_features(path, name):
    features = []
    with open(path+"\\embedding\\" + name + ".csv", 'r', encoding="utf-8") as gr:
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
    global idx
    global data

    v, g_edges, g = create_digraph(path)
    hg_edges, hg = create_hypergraph(path, '1')
    features = get_features(path, emb_type)
    data["features"].append(features)
    data["g_edge_list"].append(g_edges)
    data["hg_edge_list"].append(hg_edges)
    data["label"].append(1)
    if tag == 1:
        data["train_mask"].append(True)
        data["val_mask"].append(False)
        data["test_mask"].append(False)
    if tag == 2:
        data["train_mask"].append(False)
        data["val_mask"].append(True)
        data["test_mask"].append(False)
    if tag == 3:
        data["train_mask"].append(False)
        data["val_mask"].append(False)
        data["test_mask"].append(True)
    idx += v
    v, g_edges, g = create_digraph(path)
    features = get_features(path, emb_type)
    hg_edges1, hg1 = create_hypergraph(path, '0')
    data["features"].append(features)
    data["g_edge_list"].append(g_edges)
    data["hg_edge_list"].append(hg_edges1)
    data["label"].append(0)
    if tag == 1:
        data["train_mask"].append(True)
        data["val_mask"].append(False)
        data["test_mask"].append(False)
    if tag == 2:
        data["train_mask"].append(False)
        data["val_mask"].append(True)
        data["test_mask"].append(False)
    if tag == 3:
        data["train_mask"].append(False)
        data["val_mask"].append(False)
        data["test_mask"].append(True)
    idx += v
    # if tag == 1:
    #     train_set.loc[len(train_set)] = {'g': g, 'hg': hg, 'features': features, 'label': torch.tensor(1)}
    #     train_set.loc[len(train_set)] = {'g': g, 'hg': hg1, 'features': features, 'label': torch.tensor(0)}
    # if tag == 2:
    #     val_set.loc[len(val_set)] = {'g': g, 'hg': hg, 'features': features, 'label': torch.tensor(1)}
    #     val_set.loc[len(val_set)] = {'g': g, 'hg': hg1, 'features': features, 'label': torch.tensor(0)}
    # if tag == 3:
    #     test_set.loc[len(test_set)] = {'g': g, 'hg': hg, 'features': features, 'label': torch.tensor(1)}
    #     test_set.loc[len(test_set)] = {'g': g, 'hg': hg1, 'features': features, 'label': torch.tensor(0)}
    # if tag == 1:
    #     train_set.append([g, hg, features, torch.tensor(1)])
    #     train_set.append([g, hg1, features, torch.tensor(0)])
    # elif tag == 2:
    #     val_set.append([g, hg, features, torch.tensor(1)])
    #     val_set.append([g, hg1, features, torch.tensor(0)])
    # elif tag == 3:
    #     test_set.append([g, hg, features, torch.tensor(1)])
    #     test_set.append([g, hg1, features, torch.tensor(0)])


def scan_dir(path, tag, emb_type):
    global idx
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
    global data
    train_set.clear()
    val_set.clear()
    test_set.clear()

    print("prepare dataset")
    scan_dir('E:\\HMove\\dataset\\train', 1, emb)
    scan_dir('E:\\HMove\\dataset\\val', 2, emb)
    scan_dir('E:\\HMove\\dataset\\test', 3, emb)
    with open("data/" + emb + "/features.pkl", "wb") as f:
        pickle.dump(data["features"], f)
    with open("data/" + emb + "/g_edge_list.pkl", "wb") as f:
        pickle.dump(data["g_edge_list"], f)
    with open("data/" + emb + "/hg_edge_list.pkl", "wb") as f:
        pickle.dump(data["hg_edge_list"], f)
    with open("data/" + emb + "/labels.pkl", "wb") as f:
        pickle.dump(data["labels"], f)
    with open("data/" + emb + "/train_mask.pkl", "wb") as f:
        pickle.dump(data["train_mask"], f)
    with open("data/" + emb + "/val_mask.pkl", "wb") as f:
        pickle.dump(data["val_mask"], f)
    with open("data/" + emb + "/test_mask.pkl", "wb") as f:
        pickle.dump(data["test_mask"], f)

    print('finished')

    #return train_set, val_set, test_set

print(torch.cuda.is_available())