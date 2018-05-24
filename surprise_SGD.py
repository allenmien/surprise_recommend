# -*-coding:utf-8-*-
"""
@author:Mark
@file: surprise_SGD.py 
@time: 2018/05/24
"""
# 这里实现的算法用到的算法无外乎也是SGD等，因此也有一些超参数会影响最后的结果，
# 我们同样可以用sklearn中常用到的网格搜索交叉验证(GridSearchCV)来选择最优的参数。
# 简单的例子如下所示：

# 定义好需要优选的参数网格
from surprise import GridSearch, Dataset
from surprise.prediction_algorithms.matrix_factorization import SVD

param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}
# 使用网格搜索交叉验证
grid_search = GridSearch(SVD, param_grid, measures=['RMSE', 'FCP'])
# 在数据集上找到最好的参数
data = Dataset.load_builtin('ml-100k')
data.split(n_folds=3)
grid_search.evaluate(data)
# 输出调优的参数组
# 输出最好的RMSE结果
print(grid_search.best_score['RMSE'])
# >>> 0.96117566386

# 输出对应最好的RMSE结果的参数
print(grid_search.best_params['RMSE'])
# >>> {'reg_all': 0.4, 'lr_all': 0.005, 'n_epochs': 10}

# 最好的FCP得分
print(grid_search.best_score['FCP'])
# >>> 0.702279736531

# 对应最高FCP得分的参数
print(grid_search.best_params['FCP'])
# >>> {'reg_all': 0.6, 'lr_all': 0.005, 'n_epochs': 10}
