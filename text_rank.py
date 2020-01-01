'''
3.1 手动实现TextRank算法 (在新闻数据中随机提取100条新闻训练词向量和做做法测试）

提示：
确定窗口，建立图链接。 
通过词向量相似度确定图上边的权重
根据公式实现算法迭代(d=0.85)
'''

# 数据预处理,前面的课程已经做过词向量训练。直接读出来是df句子向量，取100条

import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')


fileHandle = open('D:/开课吧/NLP11/sen2vec_news.file', 'rb')
corp = pickle.load(fileHandle)
fileHandle.close()
corp = corp[:100]['sen2vec']
corp = corp.apply(lambda x:x[0])
corp = corp.values

# 手动实现TextRank算法
# 计算句子之间的余弦相似度，构成相似度矩阵
sim_mat = np.zeros([len(corp), len(corp)])

from sklearn.metrics.pairwise import cosine_similarity

for i in range(len(corp)):
  for j in range(len(corp)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(corp[i].reshape(1,100), corp[j].reshape(1,100))[0,0]
print("句子相似度矩阵的形状为：",sim_mat.shape)

###############################################################################
#
#迭代得到句子的textrank值，排序并取出摘要"""
import networkx as nx

# 利用句子相似度矩阵构建图结构，句子为节点，句子相似度为转移概率
nx_graph = nx.from_numpy_array(sim_mat)

# 得到所有句子的textrank值
scores = nx.pagerank(nx_graph)

# 根据textrank值对未处理的句子进行排序
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(corp)), reverse=True)

# 取出得分最高的前3个句子作为摘要
sn = 3
for i in range(sn):
    print("第"+str(i+1)+"条摘要：\n\n",ranked_sentences[i][1],'\n')
    
###############################################################################

# 根据公式实现算法迭代(d=0.85)

def textrank(self, sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'), withFlag=False):
 
    self.pos_filt = frozenset(allowPOS)
    # 定义无向有权图
    g = UndirectWeightedGraph()
    # 定义共现词典
    cm = defaultdict(int)
    # 分词
    words = tuple(self.tokenizer.cut(sentence))
    # 依次遍历每个词
    for i, wp in enumerate(words):
        # 词i 满足过滤条件
        if self.pairfilter(wp):
            # 依次遍历词i 之后窗口范围内的词
            for j in range(i + 1, i + self.span):
                # 词j 不能超出整个句子
                if j >= len(words):
                    break
                # 词j不满足过滤条件，则跳过
                if not self.pairfilter(words[j]):
                    continue
                # 将词i和词j作为key，出现的次数作为value，添加到共现词典中
                if allowPOS and withFlag:
                    cm[(wp, words[j])] += 1
                else:
                    cm[(wp.word, words[j].word)] += 1
    # 依次遍历共现词典的每个元素，将词i，词j作为一条边起始点和终止点，共现的次数作为边的权重
    for terms, w in cm.items():
        g.addEdge(terms[0], terms[1], w)
    
    # 运行textrank算法
    nodes_rank = g.rank()
    
    # 根据指标值进行排序
    if withWeight:
        tags = sorted(nodes_rank.items(), key=itemgetter(1), reverse=True)
    else:
        tags = sorted(nodes_rank, key=nodes_rank.__getitem__, reverse=True)
 
    # 输出topK个词作为关键词
    if topK:
        return tags[:topK]
    else:
        return tags
    
def addEdge(self, start, end, weight):
    # use a tuple (start, end, weight) instead of a Edge object
    self.graph[start].append((start, end, weight))
    self.graph[end].append((end, start, weight))
    
def rank(self):
    ws = defaultdict(float)
    outSum = defaultdict(float)
 
    wsdef = 1.0 / (len(self.graph) or 1.0)
    # 初始化各个结点的权值
    # 统计各个结点的出度的次数之和
    for n, out in self.graph.items():
        ws[n] = wsdef
        outSum[n] = sum((e[2] for e in out), 0.0)
 
    # this line for build stable iteration
    sorted_keys = sorted(self.graph.keys())
    # 遍历若干次
    for x in range(10):  # 10 iters
        # 遍历各个结点
        for n in sorted_keys:
            s = 0
            # 遍历结点的入度结点
            for e in self.graph[n]:
                # 将这些入度结点贡献后的权值相加
                # 贡献率 = 入度结点与结点n的共现次数 / 入度结点的所有出度的次数
                s += e[2] / outSum[e[1]] * ws[e[1]]
            # 更新结点n的权值
            ws[n] = (1 - self.d) + self.d * s
 
    (min_rank, max_rank) = (sys.float_info[0], sys.float_info[3])
 
    # 获取权值的最大值和最小值
    for w in ws:
        if w < min_rank:
            min_rank = w
        if w > max_rank:
            max_rank = w
 
    # 对权值进行归一化
    for n, w in ws.items():
        # to unify the weights, don't *100.
        ws[n] = (w - min_rank / 10.0) / (max_rank - min_rank / 10.0)
 
    return ws

import sys
from operator import itemgetter
from collections import defaultdict

 
 
class UndirectWeightedGraph:
    d = 0.85
 
    def __init__(self):
        self.graph = defaultdict(list)
 
    def addEdge(self, start, end, weight):
        # use a tuple (start, end, weight) instead of a Edge object
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))
 
    def rank(self):
        ws = defaultdict(float)
        outSum = defaultdict(float)
 
        wsdef = 1.0 / (len(self.graph) or 1.0)
        for n, out in self.graph.items():
            ws[n] = wsdef
            outSum[n] = sum((e[2] for e in out), 0.0)
 
        # this line for build stable iteration
        sorted_keys = sorted(self.graph.keys())
        for x in range(10):  # 10 iters
            for n in sorted_keys:
                s = 0
                for e in self.graph[n]:
                    s += e[2] / outSum[e[1]] * ws[e[1]]
                ws[n] = (1 - self.d) + self.d * s
 
        (min_rank, max_rank) = (sys.float_info[0], sys.float_info[3])
 
        for w in ws:
            if w < min_rank:
                min_rank = w
            if w > max_rank:
                max_rank = w
 
        for n, w in ws.items():
            # to unify the weights, don't *100.
            ws[n] = (w - min_rank / 10.0) / (max_rank - min_rank / 10.0)
 
        return ws
    
    
    