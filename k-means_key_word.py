# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 18:20:52 2020

3.2 使用词向量和k-means的方法寻找关键词

提示：
1.使用3.1训练好的词向量
2.可使用sklearn等机器学习库

@author: us
"""

import numpy as np
import pickle
import warnings
import pandas as pd
warnings.filterwarnings('ignore')


fileHandle = open('D:/开课吧/NLP11/word2vec_File.file', 'rb')
corp = pickle.load(fileHandle)
fileHandle.close()
corp = pd.DataFrame(corp)
corp = corp.iloc[:,:10000] #取10000个词做计算
from sklearn.cluster import KMeans

# weight是个矩阵，n列：n个词，每个词一列，x行：每个句子一行
weight = corp.values.T
# print data
kmeans = KMeans(n_clusters=5, random_state=0).fit(weight)#k值可以自己设置，不一定是五类
# print kmeans
centroid_list = kmeans.cluster_centers_
labels = kmeans.labels_
n_clusters_ = len(centroid_list)

# 取n个离中心点最近的词
def cosine_similarity(vector1=centroid_list[0], vector2=weight[0]):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return round(dot_product / ((normA**0.5)*(normB**0.5)), 2)

sim_list = []
for center in centroid_list:
    sim_min = 0
    for i in range(len(weight)):
        sim = cosine_similarity(center, weight[i])
        if sim > sim_min:
            sim_min = sim
            print(sim_min)
            res = i
    sim_list.append(res)



key_word = corp.columns[sim_list]
print(key_word)







