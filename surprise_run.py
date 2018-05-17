# -*-coding:utf-8-*-
"""
@author:Mark
@file: surprise_run.py 
@time: 2018/05/17
"""
from surprise import Dataset
from surprise import SVD
from surprise.model_selection import cross_validate

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

# Use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
