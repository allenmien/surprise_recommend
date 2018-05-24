# -*-coding:utf-8-*-
"""
@author:Mark
@file: surprise_music_KNNBaseline.py 
@time: 2018/05/24
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cPickle as pickle
import os

from surprise import Dataset
from surprise import Reader

# 重建歌单id到歌单名的映射字典
id_name_dic = pickle.load(open("popular_playlist.pkl", "rb"))
print("加载歌单id到歌单名的映射字典完成...")
# 重建歌单名到歌单id的映射字典
name_id_dic = {}
for playlist_id in id_name_dic:
    name_id_dic[id_name_dic[playlist_id]] = playlist_id
print("加载歌单名到歌单id的映射字典完成...")

file_path = os.path.expanduser('./popular_music_suprise_format.txt')
# 指定文件格式
reader = Reader(line_format='user item rating timestamp', sep=',')
# 从文件读取数据
music_data = Dataset.load_from_file(file_path, reader=reader)
# 计算歌曲和歌曲之间的相似度
print("构建数据集...")
trainset = music_data.build_full_trainset()
# sim_options = {'name': 'pearson_baseline', 'user_based': False}


# In[15]:

id_name_dic.keys()[2]

# In[18]:

print(id_name_dic[id_name_dic.keys()[2]])

# In[19]:

trainset.n_items

# In[20]:

trainset.n_users
