# -*-coding:utf-8-*-
"""
@author:Mark
@file: surprise_load_file.py 
@time: 2018/05/24
"""
# 指定文件所在路径
import os

from surprise import Reader, Dataset

file_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.data')
# 告诉文本阅读器，文本的格式是怎么样的
reader = Reader(line_format='user item rating timestamp', sep='\t')
# 加载数据
data = Dataset.load_from_file(file_path, reader=reader)
# 手动切分成5折(方便交叉验证)
data.split(n_folds=5)
