# -*- encoding = utf-8 -*-
"""
@description: 
@date: 2023/4/21 15:07
@File : test
@Software : PyCharm
"""
import os
import torch
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def accuracy(preds, labels):
    correct = preds.eq(labels).double()
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    correct = correct.sum()
    return correct / len(labels), p, r, f1


def solve(path):
    data = pd.read_csv(path + '\\chatgpt\\ffm_range.csv', sep=',', header=None, encoding='utf-8')
    data.columns = ['id', 'name', 'st', 'ed', 'labels', 'preds']
    labels = data['labels']
    preds = data['preds']
    labels = torch.LongTensor(labels.values)
    preds = torch.LongTensor(preds.values)
    acc, p, r, f1 = accuracy(preds, labels)
    result = list([acc.item(), p, r, f1])
    # with open(path + '\\chatgpt.txt', 'w') as cg:
    #     for x in result:
    #         cg.writelines("%.4f\n" % x)
    print("path: " + path,
          "accuracy= {:.4f}".format(acc.item()),
          "precision= {:.4f}".format(p),
          "recall= {:.4f}".format(r),
          "f1-score= {:.4f}".format(f1))


def scan_dir(path):
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isdir(file_path):
            if file_name.startswith('1'):
                solve(file_path)
            else:
                scan_dir(file_path)

# 1                                15
labels = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
preds = [1,0,1,0,0,0,0,1,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0,1,0,1,0,1,0]
# 2
labels = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
preds = [1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0,1,0,1,0,1,0,1,1,0,0,1,0,1,0,1,0]
# # 3
labels = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
preds = [0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0,0,1,1,0,1,0,0,0]
# # 4
labels = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
preds = [0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,1,1,1,1,1,1,0,1,0,1,0,1,0]
# # 5
labels = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
preds = [1,0,0,0,1,0,1,0,1,0,0,1,1,0,1,0,1,0,1,0,1,0,1,0]
# # 6
labels = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
preds = [1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,1,1,0,0,1,0,1,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0,1,0,0,1]
# # 7
labels = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
preds = [1,0,1,0,0,1,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,0,0,1,0,1,0,1,1,1,0]
# # 8
labels = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
preds = [1,0,1,0,1,0,1,1,1,0,0,1,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1]
# # 9
labels = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
preds = [0,0,0,0,1,0,1,0,0,1,1,0,1,0,1,0,1,0,1,0]
# # 10
labels = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
preds = [0,1,0,0,1,0,1,0,1,0,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,0,1,1,0]
# # 11
labels = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
preds = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1]
labels = torch.LongTensor(labels)
preds = torch.LongTensor(preds)
acc, p, r, f1 = accuracy(preds, labels)
result = list([acc.item(), p, r, f1])
# with open(path + '\\chatgpt.txt', 'w') as cg:
#     for x in result:
#         cg.writelines("%.4f\n" % x)
print("accuracy= {:.4f}".format(acc.item()),
      "precision= {:.4f}".format(p),
      "recall= {:.4f}".format(r),
      "f1-score= {:.4f}".format(f1))


