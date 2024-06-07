# -*- encoding = utf-8 -*-
"""
@description: Git instruction operation tool script
@date: 2023/3/20 8:44
@File : git_utils
@Software : PyCharm
"""

import subprocess
import os
import sys

# 根据文件名称，打印其修改历史
def logs(p, t, c):
    subprocess.Popen("git log --oneline --reverse *" + p + " > " + t,
                     cwd=os.path.dirname(c), shell=True, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE)


# 根据commit id和文件路径，获取old文件
def show(i, p, t, c):

    try:
        # 组装命令
        command = f'git show {i}:{p} > {t}'

        # 执行命令
        subprocess.Popen(command, cwd=os.path.dirname(c), shell=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print(f'Successfully executed: {command}')

    except Exception as e:
        print(f'Error: {e}')

def show1(i, p, t, c):
    subprocess.Popen("git show " + i + ":" + p + " > " + t,
                     cwd=os.path.dirname(c), shell=True,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def logs2(p, t, c):
    subprocess.Popen("git log --oneline --reverse HEAD~" + p + " > " + t,
                     cwd=os.path.dirname(c), shell=True, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE)


# argv[0]是调用的python文件路径
# op = sys.argv[1]
# path = sys.argv[2]
# to = sys.argv[3]
# cd = sys.argv[4]
# if __name__ == "__main__":
#     if op == "2":
#         id = sys.argv[5]
#         show(id, path, to, cd)
#     elif op == "1":
#         logs(path, to, cd)
#     elif op == "3":
#         logs2(path, to, cd)
#     elif op == "4":
#         id = sys.argv[5]
#         show1(id, path, to, cd)
show('00aa01fb90f3210d1e3027d7f759fb1085b814bd', 'exec/java-exec/src/test/java/org/apache/drill/exec/server/TestDrillbitResilience.java', 'E:/HMove/dataset/train/drill/TestDrillbitResilience/test.java', 'E:\\IdeaProjects\\Miner\\tmp\\drill\\')