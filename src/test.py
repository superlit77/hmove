import csv
import os


def remove_lines_from_file(file_path, start_line, end_line):
    try:
        # 打开文件并读取内容
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
            if len(lines) < end_line or lines[end_line-1].strip() != '}':
                print(f"删除过了: {file_path}")
                return
            # 删除指定行范围的代码
            # for line in lines[start_line - 1:end_line]:
            #     print(line)
            del lines[start_line - 1:end_line]

            # 将修改后的内容写回文件
            with open(file_path, 'w') as file:
                file.writelines(lines)

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return


def solve(path):
    with open(path+"/target_range.csv", 'r', encoding='utf-8') as tr:
        csvreader = csv.reader(tr)

        # 读取第一行数据
        first_row = next(csvreader)
        second_row = next(csvreader)
        print(first_row[0], path)
        # 根据逗号分割存入数组
        data_array = first_row
        second_array = second_row
        remove_lines_from_file(path+"/source/"+data_array[0]+'.java', int(second_array[0]), int(second_array[1]))


def scan_dir(path):
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isdir(file_path):
            if file_name.startswith('1') or file_name.startswith('2'):
                solve(file_path)
            else:
                scan_dir(file_path)


if __name__ == "__main__":
    scan_dir("D:\\dataset\\test")
    #solve("D:\\dataset\\test\\ant\\large\\10000")