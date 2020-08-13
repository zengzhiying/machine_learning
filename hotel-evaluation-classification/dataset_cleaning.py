#!/usr/bin/env python3
# coding=utf-8
"""酒店评论数据集清洗
转为训练集和测试集
"""

import os
import csv
import random
from pathlib import Path


RAW_DIRECTORY = 'ChnSentiCorp_htl_ba_2000'
NEG_DIR = os.path.join(RAW_DIRECTORY, 'neg')
POS_DIR = os.path.join(RAW_DIRECTORY, 'pos')

# 训练集比重 默认0.8
TRAIN_PRO = 0.8

def load_dataset(data_dir):
    """加载目录下的数据集为指定的格式
    Args:
        data_dir: 数据集目录
    Returns:
        返回: 数据集列表, 格式为: [('评论', 标签), ...]
        0 - 差评, 1 - 好评
    """
    text_files = os.listdir(data_dir)
    print(f"dir: {data_dir}, number of file: {len(text_files)}")
    datasets = []
    for text_file in text_files:
        print(text_file)
        text_path = Path(data_dir) / text_file
        with open(str(text_path), 'r', encoding='gbk', errors='ignore') as f:
            content = f.read()
        content = content.strip()
        content = content.replace('\n', ',')
        if text_file.startswith('neg'):
            datasets.append((content, 0))
        else:
            datasets.append((content, 1))

    return datasets



if __name__ == '__main__':
    neg_dataset = load_dataset(NEG_DIR)
    pos_dataset = load_dataset(POS_DIR)

    datasets = neg_dataset + pos_dataset

    # shuffle数据集 两次
    random.shuffle(datasets)
    random.shuffle(datasets)

    # 切分
    train_size = int(len(datasets) * TRAIN_PRO)

    train_dataset, test_dataset = datasets[:train_size], datasets[train_size:]

    # 保存
    with open('hotel_train.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(train_dataset)

    with open('hotel_test.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(test_dataset)


