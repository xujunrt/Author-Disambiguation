#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-12-11 下午7:09
# @Author  : xujun
# @Email   : xujunrt@163.com
# @Version : 1.0
import os
import math
import numpy as np

# 自定义sigmoid函数
def sigmoid(x):
    # return max(0.0001,float(x))
    # return x
    # return float(x) /(1+math.exp(-x))

    return float(1) / (1 + math.exp(-x))


    # return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

# 构建关于文章的矩阵
def construct_doc_matrix(dict, paper_list):
    """
    construct the learned embedding for document clustering
    dict: {paper_index, numpy_array}
    """
    D_matrix = dict[paper_list[0]]
    for idx in xrange(1, len(paper_list)):
        D_matrix = np.vstack((D_matrix, dict[paper_list[idx]]))
    return D_matrix


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if len(x)>0:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    else:
        return 0.5


def save_embedding(dict, paper_list, num_dimen,filename):
    """
    save the final embedding results for each document
    """
    fn = filename.split('/')[-1]
    embedding_file = open('/home/xujun/project/disambiguation_embedding/emb/'+fn+'.txt','w')
    # embedding_file.write(str(len(paper_list)) + ' ' + str(num_dimen) + os.linesep)
    D_matrix = dict[paper_list[0]]
    for idx in xrange(1, len(paper_list)):
        D_matrix = np.vstack((D_matrix, dict[paper_list[idx]]))
    D_matrix = np.hstack((np.array([range(1, len(paper_list)+1)]).T, D_matrix))
    np.savetxt(embedding_file, D_matrix[:,1:],
               fmt = ' '.join(['%1.5f'] * num_dimen))
