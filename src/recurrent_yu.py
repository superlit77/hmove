import re
import csv
import os
import random
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

all_id = 0
all_classid = 0


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        return content


def calculate_cyclomatic_complexity(codes, st, ed):
    code_slice = codes[st:ed + 1]
    code_text = '\n'.join(code_slice)

    # Count the occurrences of control flow structures
    if_count = len(re.findall(r'\bif\b', code_text))
    for_count = len(re.findall(r'\bfor\b', code_text))
    while_count = len(re.findall(r'\bwhile\b', code_text))
    case_count = len(re.findall(r'\bcase\b', code_text))
    catch_count = len(re.findall(r'\bcatch\b', code_text))
    ternary_count = len(re.findall(r'\?', code_text))

    # Calculate cyclomatic complexity
    complexity = if_count + for_count + while_count + case_count + catch_count + ternary_count + 1

    return complexity


def count_method_parameters(codes, st):
    param_pattern = r'\((.*?)\)'
    params = re.findall(param_pattern, codes[st])
    #print(params)
    # 计算逗号数量
    comma_count = codes[st].count(',')
    # 判断是否有参数
    if len(params) == 1 and params[0] == '':
        return 0
    return comma_count + 1


def count_method_calls(id, path):
    ans = 0

    # 判断to是不是method
    def judge(to):
        with open(path + "\\graph_node.csv", 'r', encoding="utf-8") as gr:
            for line in gr:
                line = line.strip('\n').split(',')
                no_id = int(line[0])
                no_tag = int(line[7])
                if no_id == to and no_tag == 2:
                    return True
        return False
    with open(path + "\\relation.csv", 'r', encoding="utf-8") as gr:
        for line in gr:
            line = line.strip('\n').split(',')
            fr = int(line[0])
            to = int(line[1])
            # fr 调用 to
            if fr == id and judge(to):
                ans += 1
    return ans


def count_in_external_method_calls(id, path, source_class):
    ans_e = 0
    ans_i = 0
    external_method = []
    def judge_external(to):
        with open(path + "\\graph_node.csv", 'r', encoding="utf-8") as gr:
            for line in gr:
                line = line.strip('\n').split(',')
                no_id = int(line[0])
                no_class = line[2].split('-')[0]
                no_tag = int(line[7])
                if no_id == to and no_class != source_class and no_tag == 2:
                    if no_id not in external_method:
                        external_method.append(no_id)
                    return True
        return False

    def judge_internal(to):
        with open(path + "\\graph_node.csv", 'r', encoding="utf-8") as gr:
            for line in gr:
                line = line.strip('\n').split(',')
                no_id = int(line[0])
                no_class = line[2].split('-')[0]
                no_tag = int(line[7])
                if no_id == to and no_class == source_class and no_tag == 2:
                    return True
        return False

    with open(path + "\\relation.csv", 'r', encoding="utf-8") as gr:
        for line in gr:
            line = line.strip('\n').split(',')
            fr = int(line[0])
            to = int(line[1])
            # judge 判断method fr 和 to是不是属于不同类
            if fr == id and judge_external(to):
                ans_e += 1
            # judge 判断method fr 和 to是不是属于同一类
            if fr == id and judge_internal(to):
                ans_i += 1
    return ans_e, ans_i, external_method


def count_external_class_calls(id, path, source_class):
    classes = {}

    def judge_external(to):
        with open(path + "\\graph_node.csv", 'r', encoding="utf-8") as gr:
            for line in gr:
                line = line.strip('\n').split(',')
                no_id = int(line[0])
                no_class = line[2].split('-')[0]
                if no_id == to and no_class != source_class:
                    return no_class
        return ''

    with open(path + "\\relation.csv", 'r', encoding="utf-8") as gr:
        for line in gr:
            line = line.strip('\n').split(',')
            fr = int(line[0])
            to = int(line[1])
            # judge 判断method fr 和 to是不是属于不同类
            find_class = judge_external(to)
            if fr == id and find_class != '':
                if find_class in classes:
                    classes[find_class] += 1
                else:
                    classes[find_class] = 1
    if len(classes) == 0:
        return 0, 0
    return len(classes), max(classes.values())


def solve(path):
    global all_id
    global all_classid
    datas = pd.DataFrame({
        'id': pd.Series([], dtype=np.int64),
        'class-id': pd.Series([], dtype='str'),
        'loc': pd.Series([], dtype=np.int64),
        'cc': pd.Series([], dtype=np.int64),
        'pc': pd.Series([], dtype=np.int64),
        'ncm': pd.Series([], dtype=np.int64),
        'ncmec': pd.Series([], dtype=np.int64),
        'ncmic': pd.Series([], dtype=np.int64),
        'neca': pd.Series([], dtype=np.int64),
        'namfaec': pd.Series([], dtype=np.int64),
        'label': pd.Series([], dtype=np.int64)
    }, columns=['id', 'class-id', 'loc', 'cc', 'pc', 'ncm', 'ncmec', 'ncmic', 'neca', 'namfaec', 'label'])

    source_id = -1
    if not os.path.exists(path + "\\relation.csv"):
        open(path + "\\relation.csv", 'w').close()
    with open(path + "\\graph_node.csv", 'r', encoding="utf-8") as gr:
        for line in gr:
            line = line.strip('\n').split(',')
            no_tag = line[8]
            if no_tag == "1":
                source_file = line[1]
                source_class = line[2].split('-')[0]
                source_id = int(line[0])
                break
    #print(path, source_file, source_class, source_id)

    def go(st, ed, ids, codes):
        print(path, st, ed, ids)
        # 代码行
        global all_id
        global all_classid
        loc = ed - st + 1
        # 圈复杂度
        cc = calculate_cyclomatic_complexity(codes, st, ed)
        # 参数数量
        pc = count_method_parameters(codes, st)
        # 调用方法的次数
        ncm = count_method_calls(ids, path)
        # 对外部类方法的调用次数/对内部类方法的调用次数
        ncmec, ncmic, ex_method = count_in_external_method_calls(ids, path, source_class)
        # 访问的外部类的数量/访问最常访问的外部类的次数
        neca, namfaec = count_external_class_calls(ids, path, source_class)
        # label
        label = 1 if ids == source_id else 0
        new_row = {'id': all_id, 'class-id': all_classid, 'loc': loc, 'cc': cc, 'pc': pc, 'ncm': ncm, 'ncmec': ncmec,
                   'ncmic': ncmic,
                   'neca': neca, 'namfaec': namfaec, 'label': label}
        #print(new_row)
        new_id = all_id
        all_id += 1
        return new_row, set(ex_method), new_id

    if source_id == -1:
        return

    ex_method_ids = set()
    source_ids = set()
    source_ids.add(source_id)
    id_to_new = dict()
    java_file = read_file(path + "\\source\\" + source_file)
    code = java_file.split('\n')
    codes = [''] + code

    def internal(fr):
        with open(path + "\\graph_node.csv", 'r', encoding="utf-8") as gr:
            for line in gr:
                line = line.strip('\n').split(',')
                no_id = int(line[0])
                no_class = line[2].split('-')[0]
                no_tag = int(line[7])
                if no_id == fr and no_class == source_class and no_tag == 2:
                    return True
        return False

    with open(path + "\\relation.csv", 'r', encoding="utf-8") as gr:
        for line in gr:
            line = line.strip('\n').split(',')
            fr = int(line[0])
            to = int(line[1])
            # judge 判断method fr 和 to是不是属于同类
            if internal(fr) and len(source_ids) < 5:
                source_ids.add(fr)
            if internal(to) and len(source_ids) < 5:
                source_ids.add(to)

    with open(path + "\\graph_node.csv", 'r', encoding="utf-8") as gr:
        for line in gr:
            line = line.strip('\n').split(',')
            no_id = int(line[0])
            no_file = line[1]
            if no_id in source_ids:
                start_line = int(line[5])
                end_line = int(line[6])
                print(all_id, no_file, 1)
                datas.loc[len(datas)], ex_ids, new_id = go(start_line, end_line, no_id, codes)
                id_to_new[no_id] = new_id
                ex_method_ids |= ex_ids
    pre_class = ''
    # 访问的外部method也要参与计算
    with open(path + "\\graph_node.csv", 'r', encoding="utf-8") as gr:
        for line in gr:
            line = line.strip('\n').split(',')
            no_id = int(line[0])
            no_class = line[2].split('-')[0]
            no_file = line[1]
            if no_id in ex_method_ids:
                java_file = read_file(path + "\\source\\" + no_file)
                code = java_file.split('\n')
                codet = [''] + code
                if pre_class == '':
                    all_classid += 1
                    pre_class = no_class
                elif pre_class != no_class:
                    all_classid += 1
                    pre_class = no_class
                start_line = int(line[5])
                end_line = int(line[6])
                print(all_id, no_file, 2)
                datas.loc[len(datas)], _, new_id = go(start_line, end_line, no_id, codet)
                id_to_new[no_id] = new_id

    all_classid += 1
    source_ids |= ex_method_ids
    if all_id >= 0:
        with open(path + "\\relation.csv", 'r', encoding="utf-8") as gr:
            for line in gr:
                line = line.strip('\n').split(',')
                fr = int(line[0])
                to = int(line[1])
                if fr in source_ids and to in source_ids:
                    with open('cora.cites', mode='a', newline='') as file:
                        writer = csv.writer(file, delimiter=' ')
                        writer.writerow([id_to_new[fr], id_to_new[to]])
        with open('cora.content', mode='a', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            for i in range(len(datas)):
                row = datas.iloc[i]
                writer.writerow(row)
    # datas.to_csv(path + '\\recurrent_yu.csv', sep=',', header=False, index=False, encoding='utf-8')
    # datas.drop(datas.index, inplace=True)

fst_val = 0
fst_test = 0

def scan_dir(path):
    global fst_val
    global fst_test
    global all_id

    for file_name in os.listdir(path):
        if file_name == 'zzz_val':
            print('val_id', all_id)
            fst_val = 1
        if file_name == 'zzz_ztest':
            print('test_id', all_id)
            fst_test = 1
        file_path = os.path.join(path, file_name)
        if os.path.isdir(file_path):
            if file_name.startswith('1') or file_name.startswith('2'):
                solve(file_path)
            else:
                scan_dir(file_path)


scan_dir('E:\\HMove\\dataset\\train')
