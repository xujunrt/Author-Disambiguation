#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-12-11 下午7:09
# @Author  : xujun
# @Email   : xujunrt@163.com
# @Version : 1.0

from sklearn.cluster import *
from sklearn.decomposition import *
from sklearn.externals import joblib
import numpy as np
from utility import construct_doc_matrix
from xmeans import XMeans
from ntee.model_reader import ModelReader
import hdbscan
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def hdbscan(D_matrix):
    return hdbscan.HDBSCAN(min_cluster_size=1).fit_predict(D_matrix)
def dbscan(D_matrix):
    return DBSCAN(eps=1.5, min_samples=2).fit_predict(D_matrix)
def ap(D_matrix):
    return AffinityPropagation(damping=0.6).fit_predict(D_matrix)
def xmeans(D_matrix):
    return XMeans(random_state=1).fit_predict(D_matrix)
def meanshift(D_matrix):
    # bandwidth=estimate_bandwidth(D_matrix,quantile=0.4)
    return MeanShift().fit_predict(D_matrix)
def kmeans(D_matrix,true_cluster_size):
    return KMeans(n_clusters=true_cluster_size,init="k-means++").fit_predict(D_matrix)
def hac(D_matrix,true_cluster_size):
    return AgglomerativeClustering(n_clusters = true_cluster_size,linkage = "average",affinity = "cosine").fit_predict(D_matrix)
def compute_distance(D_matrix,predict_label_dict):
    avg_distance = 0
    sum_list = []
    for key, value in predict_label_dict.iteritems():
        list_item = []
        for j in value:
            list_item.append(D_matrix[j])
        D_itme_array = np.array(list_item)
        for i in range(D_itme_array.shape[0]):
            sum_distance = 0
            for j in range(D_itme_array.shape[0]):
                dist = np.sqrt(np.sum(np.square(D_itme_array[i] - D_itme_array[j])))
                sum_distance += dist
            sum_list.append(sum_distance)
        avg_distance = float(np.median(sum_list)) / D_itme_array.shape[0]
        break
    return avg_distance
def compute_sd(D_matrix,predict_label_dict):
    D_matrix = D_matrix
    dstd = np.std(D_matrix)
    dvar = D_matrix.shape[0]*dstd*dstd
    sum_ci_var = 0.0
    center = []
    for key ,val in predict_label_dict.iteritems():
        C_i_matrix = D_matrix[0]
        for i in val:
            C_i_matrix=np.row_stack((C_i_matrix,D_matrix[i-1]))
        np.delete(C_i_matrix,0,axis=0)
        c_i_std = np.std(C_i_matrix)
        c_i_var = C_i_matrix.shape[0]*c_i_std*c_i_std
        sum_ci_var+=c_i_std
        center.append(np.sum(C_i_matrix,axis=0)/float(C_i_matrix.shape[0]))
    scat = sum_ci_var/float(len(predict_label_dict)*dvar)
    return scat

class Evaluator():
    @staticmethod
    def compute_f1(dataset, bpr_optimizer):
        D_matrix = construct_doc_matrix(bpr_optimizer.paper_latent_matrix,dataset.paper_list)
        # X_embedded = TSNE(n_components=2).fit_transform(D_matrix)
        # x = []
        # y = []
        # for i in range(X_embedded.shape[0]):
        #     x.append(X_embedded[i][0])
        #     y.append(X_embedded[i][1])
        # plt.scatter(x, y)
        # plt.show()
        # print x
        # print y
        true_cluster_size = len(set(dataset.label_list))
        true_label_dict = {}
        for idx, true_lbl in enumerate(dataset.label_list):
            if true_lbl not in true_label_dict:
                true_label_dict[true_lbl] = [idx]
            else:
                true_label_dict[true_lbl].append(idx)
        predict_label_dict = {}
        y_pred = dbscan(D_matrix)
        for idx, pred_lbl in enumerate(y_pred):
            if pred_lbl not in predict_label_dict:
                predict_label_dict[pred_lbl] = [idx]
            else:
                predict_label_dict[pred_lbl].append(idx)

        # y_pred1 = ap(D_matrix)
        # predict_label_dict1 = {}
        # for idx, pred_lbl in enumerate(y_pred1):
        #     if pred_lbl not in predict_label_dict1:
        #         predict_label_dict1[pred_lbl] = [idx]
        #     else:
        #         predict_label_dict1[pred_lbl].append(idx)
        # avg_distance1 = compute_distance(D_matrix,predict_label_dict1)
        # sd1 = compute_sd(D_matrix,predict_label_dict1)
        # print sd1, "dbscan 距离"
        # print avg_distance1,"距离"
        #
        # y_pred2 = xmeans(D_matrix)
        # predict_label_dict2={}
        # for idx, pred_lbl in enumerate(y_pred2):
        #     if pred_lbl not in predict_label_dict2:
        #         predict_label_dict2[pred_lbl] = [idx]
        #     else:
        #         predict_label_dict2[pred_lbl].append(idx)
        # sd2 = compute_sd(D_matrix, predict_label_dict2)
        # print sd2, "ap 距离"
        # predict_label_dict={}
        # #if avg_distance1>1.75:
        # if sd2>sd1 and avg_distance1>1.7:
        #     predict_label_dict=predict_label_dict2
        # else:
        #     predict_label_dict = predict_label_dict1


        # print predict_label_dict
        # print true_label_dict
        # compute cluster-level F1
        # let's denote C(r) as clustering result and T(k) as partition (ground-truth)
        # construct r * k contingency table for clustering purpose
        r_k_table = []
        for v1 in predict_label_dict.itervalues():
            k_list = []
            for v2 in true_label_dict.itervalues():
                N_ij = len(set(v1).intersection(v2))
                k_list.append(N_ij)
            r_k_table.append(k_list)
        r_k_matrix = np.array(r_k_table)
        r_num = int(r_k_matrix.shape[0])
        # compute F1 for each row C_i
        print r_k_table
        sum_f1 = 0.0
        sum_pre = 0.0
        sum_rec = 0.0
        for row in xrange(0, r_num):
            row_sum = np.sum(r_k_matrix[row,:])
            if row_sum != 0:
                max_col_index = np.argmax(r_k_matrix[row,:])
                row_max_value = r_k_matrix[row, max_col_index]
                prec = float(row_max_value) / row_sum
                col_sum = np.sum(r_k_matrix[:, max_col_index])
                rec = float(row_max_value) / col_sum
                row_f1 = float(2 * prec * rec) / (prec + rec)
                sum_f1 += row_f1
                sum_pre += prec
                sum_rec += rec
        # print len(y_pred)
        average_f1 =float(sum_f1) / r_num
        #average_f1 = float(sum_f1)
        average_pre = float(sum_pre) /r_num
        average_rec = float(sum_rec) /r_num
        return average_f1,average_pre,average_rec