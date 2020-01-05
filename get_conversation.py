# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 18:33:27 2020

3.3 提取新闻人物里的对话。(使用以上提取小数据即可）

提示：
1.寻找预料里具有表示说的意思。近义词
2.使用语法分析提取句子结构。pyltp
3.检测谓语是否有表示说的意思。

@author: us
"""
import pyltp
from gensim.models import Word2Vec
from pyltp import Segmentor
import jieba
from gensim.models.word2vec import LineSentence
from pyltp import  SentenceSplitter,NamedEntityRecognizer,Postagger,Parser,Segmentor
from gensim import models
import numpy as np
import pandas as pd

cws_model = "E:/my_path/NLP12/ltp_data_v3.4.0/cws.model"
pos_model = "E:/my_path/NLP12/ltp_data_v3.4.0/pos.model"
par_model = "E:/my_path/NLP12/ltp_data_v3.4.0/parser.model"
ner_model = "E:/my_path/NLP12/ltp_data_v3.4.0/ner.model"


def get_word_list(sentence,model):
    #得到分词
    segmentor = Segmentor()
    segmentor.load(model)
    word_list = list(segmentor.segment(sentence))
    segmentor.release()
    return word_list

def get_postag_list(word_list,model):
    #得到词性标注
    postag = Postagger()
    postag.load(model)
    postag_list = list(postag.postag(word_list))
    postag.release()
    return postag_list

def get_parser_list(word_list,postag_list,model):
    #得到依存关系
    parser = Parser()
    parser.load(model)
    arcs = parser.parse(word_list,postag_list)
    arc_list = [(arc.head,arc.relation) for arc in arcs]
    parser.release()
    return arc_list

df_cor = pd.read_csv('C:/Users/us/Desktop/sqlResult_1558435.csv')
df_cor = df_cor[['content']][0:100]
sentence = df_cor.values.flatten()

news = ''
for x in sentence:
    news = news + x

news = '''
古川俊太郎说并不想让公众抱有一种误解“任天堂对新技术置之不理”，相反，任天堂内部不断在研究和开发。
硬件开发团队评估到目前为止可用的各种新技术，随后和软件团队进行协商。
古川俊太郎还特意提到了增强现实也就是AR技术，称其绝对为最感兴趣的方向之一。
随后其谈到了逐渐成为风口浪尖的云游戏。
古川俊太郎认为云游戏可能会在10年内发展十分迅猛，但目前并不觉得云游戏会影响到主机的发展。
同时他承认仅仅关注在独占平台游戏的市场方法将会变得毫无意义，因为日后一旦玩家发现能够在其他平台甚至手机上串流任何游戏，坚持独守平台就会完蛋。
因此在任天堂真正得出结果之前还有很长一段路要走。
最后古川俊太郎笑言自己最喜欢的Switch新发售游戏是《宝可梦：剑/盾》，因为他个人十分享受收集/培育精灵的过程。

百度说早在2012年就建立了深度学习研究院，此后陆续建立了大数据实验室及硅谷实验室。小明觉得很好吃。小红说好开心。
'''


   
sents = SentenceSplitter.split(news)
sents_list = list(sents)
for i in sents_list:
    if i == '':
        sents_list.remove('')
    
#sentence = news
#model = cws_model
'''
word_list = get_word_list(news,cws_model)

# word_list = [word for word in jieba.cut(sentence)]

postag_list = get_postag_list(word_list,pos_model)

parser_list = get_parser_list(word_list,postag_list,par_model)

for i in range(len(word_list)):
    print(word_list[i],parser_list[i])
'''

# 筛选出'说'或者 与'说'相近的词，如果是谓语，找出这句话，还有这句话的主语，分割'说'后面 内容
########################### 找出与 说 意思相近的词 ################################
'''
from gensim.models import word2vec

def my_word2vec(cut_filename):
    mysetence = word2vec.Text8Corpus(cut_filename)
    # model = word2vec.Word2Vec(mysetence, size=300, min_count=1, window=5, hs=5)
    model = word2vec.Word2Vec(mysetence, size=100, min_count=1, window=5, hs=5)
    # model.save('D:/开课吧/project1/model/zh_wiki_global.model')
    return model

model = my_word2vec('D:/开课吧/project1/wiki_cut/wiki_001.txt')


for key in model.similar_by_word(u'说', topn=10):
        print(key)

('说道', 0.7895047664642334)
('问', 0.7791535258293152)
('称赞', 0.7396500706672668)
('回答', 0.7354450821876526)
('写道', 0.7224162817001343)
('所说', 0.7162884473800659)
('名言', 0.7127034068107605)
('时说', 0.7080138921737671)
('感叹', 0.7067703008651733)
('断言', 0.6969820857048035)
'''
######################### 找出 说 是 谓语 的情况 #################################

say_list = ['说', '表示', '指出', '觉得', '认为', '表明', '说道','笑言','断言']
'''
for i in range(len(word_list)):
    if (word_list[i] in say_list) :
        print(word_list[i],parser_list[i])
'''

######################### 找出主语，找出 说 后面的句子 #############################
'''
for i in range(len(word_list)):
    if (parser_list[i][1] == 'SBV') and (word_list[i+1] in say_list) :
        print(word_list[i],word_list[i+1])
''' 
      
for s in range(len(sents_list)):
    word_list_s = get_word_list(sents_list[s],cws_model)
    postag_list_s = get_postag_list(word_list_s,pos_model)
    parser_list_s = get_parser_list(word_list_s,postag_list_s,par_model)
    for i in range(len(word_list_s)):        
        if (parser_list_s[i][1] == 'SBV') and (word_list_s[i+1] in say_list) :
            print(sents_list[s])
            continue










