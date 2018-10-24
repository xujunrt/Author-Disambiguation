#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-12-11 下午7:09
# @Author  : xujun
# @Email   : xujunrt@163.com
# @Version : 1.0
import numpy as np
from utility import sigmoid

class BprOptimizer():
    """
    use Bayesian Personalized Ranking for objective loss
    latent_dimen: latent dimension
    alpha: learning rate
    matrix_reg: regularization parameter of matrix
    """
    def __init__(self, latent_dimen, alpha, matrix_reg):
        self.latent_dimen = latent_dimen
        self.alpha = alpha
        self.matrix_reg = matrix_reg

    def get_merge(self, dataset):
        node_dict = {}
        for node in dataset.Abstract_Graph.nodes():
            neighbors_list = dataset.Abstract_Graph.neighbors(node)
            if len(neighbors_list) != 1:
                node_dict[node] = []
                for item in neighbors_list:
                    if len(dataset.Abstract_Graph.neighbors(item)) == 1:
                        node_dict[node].append(item)
        print node_dict
        return node_dict

    def init_model(self, dataset):
        """
        initialize matrix using uniform [-0.2, 0.2]
        """
        self.paper_latent_matrix = {}
        self.author_latent_matrix = {}
        self.title_latent_matrix = {}
        self.jconf_latent_matrix = {}
        self.org_latent_matrix = {}
        self.year_latent_matrix = {}
        self.abstract_latent_matrix = {}
        # dd,dt,djconf,dorg,dyear,dabstract
        self.w=[1,1,0.1,0.1,0.1,1]
        node_dict = self.get_merge(dataset)
        in_list = []
        for key,val in node_dict.iteritems():
            in_list=in_list+[key]
            in_list=in_list+val
            in_list=list(set(in_list))
        list_out = list(set(dataset.paper_list)-set(in_list))
        for item in list_out:
            self.paper_latent_matrix[item] = np.random.uniform(-0.5, 0.5,self.latent_dimen)
        for key ,val in node_dict.iteritems():
            item_vec = np.random.uniform(-0.5, 0.5,self.latent_dimen)
            self.paper_latent_matrix[key] = item_vec
            for item in val:
                    self.paper_latent_matrix[item] = item_vec

        # for paper_idx in dataset.paper_list:
        #     self.paper_latent_matrix[paper_idx] = np.random.uniform(-0.5, 0.5,
        #                                                             self.latent_dimen)
        for author_idx in dataset.coauthor_list:
            self.author_latent_matrix[author_idx] = np.random.uniform(-0.5, 0.5,
                                                                      self.latent_dimen)
        for title_idx in dataset.paper_title:
            self.title_latent_matrix[title_idx] = np.random.uniform(-0.5, 0.5,
                                                                      self.latent_dimen)
        for jconf_idx in dataset.paper_jconf:
            self.jconf_latent_matrix[jconf_idx] = np.random.uniform(-0.5, 0.5,
                                                                      self.latent_dimen)
        for org_idx in dataset.paper_org:
            self.org_latent_matrix[org_idx] = np.random.uniform(-0.5, 0.5,
                                                                      self.latent_dimen)
        for year_idx in dataset.paper_year:
            self.year_latent_matrix[year_idx] = np.random.uniform(-0.5, 0.5,
                                                                      self.latent_dimen)
        for abstract_idx in dataset.paper_abstract:
            self.abstract_latent_matrix[abstract_idx] = np.random.uniform(-0.5, 0.5,
                                                                      self.latent_dimen)

    def update_pp_gradient(self, fst, snd, third):
        """
        SGD inference
        """
        x = self.predict_score(fst, snd, "pp") - \
            self.predict_score(fst, third, "pp")
        common_term = sigmoid(x) - 1

        grad_fst = common_term * (self.author_latent_matrix[snd] - \
                                  self.author_latent_matrix[third]) + \
                    2 * self.matrix_reg * self.author_latent_matrix[fst]
        self.author_latent_matrix[fst] = self.author_latent_matrix[fst] - \
                                         self.alpha * grad_fst

        grad_snd = common_term * self.author_latent_matrix[fst] + \
                   2 * self.matrix_reg * self.author_latent_matrix[snd]
        self.author_latent_matrix[snd]= self.author_latent_matrix[snd] - \
                                        self.alpha * grad_snd

        grad_third = -common_term * self.author_latent_matrix[fst] + \
                     2 * self.matrix_reg * self.author_latent_matrix[third]
        self.author_latent_matrix[third] = self.author_latent_matrix[third] - \
                                           self.alpha * grad_third

    def update_pd_gradient(self, fst, snd, third):
        x = self.predict_score(fst, snd, "pd") - \
            self.predict_score(fst, third, "pd")
        common_term = sigmoid(x) - 1

        grad_fst = common_term * (self.author_latent_matrix[snd] - \
                                  self.author_latent_matrix[third]) + \
                   2 * self.matrix_reg * self.paper_latent_matrix[fst]
        self.paper_latent_matrix[fst] = self.paper_latent_matrix[fst] - \
                                         self.alpha * grad_fst

        grad_snd = common_term * self.paper_latent_matrix[fst] + \
                   2 * self.matrix_reg * self.author_latent_matrix[snd]
        self.author_latent_matrix[snd]= self.author_latent_matrix[snd] - \
                                        self.alpha * grad_snd

        grad_third = -common_term * self.paper_latent_matrix[fst] + \
                     2 * self.matrix_reg * self.author_latent_matrix[third]
        self.author_latent_matrix[third] = self.author_latent_matrix[third] - \
                                           self.alpha * grad_third

    def update_dd_gradient(self, fst, snd, third):
        x = self.predict_score(fst, snd, "dd") - \
            self.predict_score(fst, third, "dd")
        common_term = sigmoid(x) - 1

        grad_fst = common_term * (self.paper_latent_matrix[snd] - \
                                  self.paper_latent_matrix[third]) + \
                   2 * self.matrix_reg * self.paper_latent_matrix[fst]
        self.paper_latent_matrix[fst] = self.paper_latent_matrix[fst] - \
                                         self.alpha * grad_fst * self.w[0]

        grad_snd = common_term * self.paper_latent_matrix[fst] + \
                   2 * self.matrix_reg * self.paper_latent_matrix[snd]
        self.paper_latent_matrix[snd]= self.paper_latent_matrix[snd] - \
                                       self.alpha * grad_snd* self.w[0]

        grad_third = -common_term * self.paper_latent_matrix[fst] + \
                     2 * self.matrix_reg * self.paper_latent_matrix[third]
        self.paper_latent_matrix[third] = self.paper_latent_matrix[third] - \
                                          self.alpha * grad_third* self.w[0]
    def update_dt_gradient(self, fst, snd, third):
        x = self.predict_score(fst, snd, "dt") - \
            self.predict_score(fst, third, "dt")
        common_term = sigmoid(x) - 1

        grad_fst = common_term * (self.title_latent_matrix[snd] - \
                                  self.title_latent_matrix[third]) + \
                   2 * self.matrix_reg * self.paper_latent_matrix[fst]
        self.paper_latent_matrix[fst] = self.paper_latent_matrix[fst] - \
                                         self.alpha * grad_fst* self.w[1]

        grad_snd = common_term * self.paper_latent_matrix[fst] + \
                   2 * self.matrix_reg * self.title_latent_matrix[snd]
        self.title_latent_matrix[snd]= self.title_latent_matrix[snd] - \
                                       self.alpha * grad_snd* self.w[1]

        grad_third = -common_term * self.paper_latent_matrix[fst] + \
                     2 * self.matrix_reg * self.title_latent_matrix[third]
        self.title_latent_matrix[third] = self.title_latent_matrix[third] - \
                                          self.alpha * grad_third* self.w[1]

    def update_djconf_gradient(self, fst, snd, third):
        x = self.predict_score(fst, snd, "djconf") - \
            self.predict_score(fst, third, "djconf")
        common_term = sigmoid(x) - 1

        grad_fst = common_term * (self.jconf_latent_matrix[snd] - \
                                  self.jconf_latent_matrix[third]) + \
                   2 * self.matrix_reg * self.paper_latent_matrix[fst]
        self.paper_latent_matrix[fst] = self.paper_latent_matrix[fst] - \
                                         self.alpha * grad_fst* self.w[2]

        grad_snd = common_term * self.paper_latent_matrix[fst] + \
                   2 * self.matrix_reg * self.jconf_latent_matrix[snd]
        self.jconf_latent_matrix[snd]= self.jconf_latent_matrix[snd] - \
                                       self.alpha * grad_snd* self.w[2]

        grad_third = -common_term * self.paper_latent_matrix[fst] + \
                     2 * self.matrix_reg * self.jconf_latent_matrix[third]
        self.jconf_latent_matrix[third] = self.jconf_latent_matrix[third] - \
                                          self.alpha * grad_third* self.w[2]

    def update_dorg_gradient(self, fst, snd, third):
        x = self.predict_score(fst, snd, "dorg") - \
            self.predict_score(fst, third, "dorg")
        common_term = sigmoid(x) - 1

        grad_fst = common_term * (self.org_latent_matrix[snd] - \
                                  self.org_latent_matrix[third]) + \
                   2 * self.matrix_reg * self.paper_latent_matrix[fst]
        self.paper_latent_matrix[fst] = self.paper_latent_matrix[fst] - \
                                        self.alpha * grad_fst* self.w[3]

        grad_snd = common_term * self.paper_latent_matrix[fst] + \
                   2 * self.matrix_reg * self.org_latent_matrix[snd]
        self.org_latent_matrix[snd] = self.org_latent_matrix[snd] - \
                                        self.alpha * grad_snd* self.w[3]

        grad_third = -common_term * self.paper_latent_matrix[fst] + \
                     2 * self.matrix_reg * self.org_latent_matrix[third]
        self.org_latent_matrix[third] = self.org_latent_matrix[third] - \
                                          self.alpha * grad_third* self.w[3]

    def update_dyear_gradient(self, fst, snd, third):
        x = self.predict_score(fst, snd, "dyear") - \
            self.predict_score(fst, third, "dyear")
        common_term = sigmoid(x) - 1

        grad_fst = common_term * (self.year_latent_matrix[snd] - \
                                  self.year_latent_matrix[third]) + \
                   2 * self.matrix_reg * self.paper_latent_matrix[fst]
        self.paper_latent_matrix[fst] = self.paper_latent_matrix[fst] - \
                                        self.alpha * grad_fst* self.w[4]

        grad_snd = common_term * self.paper_latent_matrix[fst] + \
                   2 * self.matrix_reg * self.year_latent_matrix[snd]
        self.year_latent_matrix[snd] = self.year_latent_matrix[snd] - \
                                      self.alpha * grad_snd* self.w[4]

        grad_third = -common_term * self.paper_latent_matrix[fst] + \
                     2 * self.matrix_reg * self.year_latent_matrix[third]
        self.year_latent_matrix[third] = self.year_latent_matrix[third] - \
                                        self.alpha * grad_third* self.w[4]
    def update_dabstract_gradient(self, fst, snd, third):
        x = self.predict_score(fst, snd, "dabstract") - \
            self.predict_score(fst, third, "dabstract")
        common_term = sigmoid(x) - 1

        grad_fst = common_term * (self.abstract_latent_matrix[snd] - \
                                  self.abstract_latent_matrix[third]) + \
                   2 * self.matrix_reg * self.paper_latent_matrix[fst]
        self.paper_latent_matrix[fst] = self.paper_latent_matrix[fst] - \
                                        self.alpha * grad_fst* self.w[5]

        grad_snd = common_term * self.paper_latent_matrix[fst] + \
                   2 * self.matrix_reg * self.abstract_latent_matrix[snd]
        self.abstract_latent_matrix[snd] = self.abstract_latent_matrix[snd] - \
                                        self.alpha * grad_snd* self.w[5]

        grad_third = -common_term * self.paper_latent_matrix[fst] + \
                     2 * self.matrix_reg * self.abstract_latent_matrix[third]
        self.abstract_latent_matrix[third] = self.abstract_latent_matrix[third] - \
                                          self.alpha * grad_third* self.w[5]

    def compute_pp_loss(self, fst, snd, third):
        """
        loss includes ranking loss and model complexity
        """
        x = self.predict_score(fst, snd, "pp") - \
             self.predict_score(fst, third, "pp")
        ranking_loss = -np.log(sigmoid(x))

        complexity = 0.0
        complexity += self.matrix_reg * np.dot(self.author_latent_matrix[fst],
                                               self.author_latent_matrix[fst])
        complexity += self.matrix_reg * np.dot(self.author_latent_matrix[snd],
                                               self.author_latent_matrix[snd])
        complexity += self.matrix_reg * np.dot(self.author_latent_matrix[third],
                                               self.author_latent_matrix[third])
        return ranking_loss + complexity

    def compute_pd_loss(self, fst, snd, third):
        x = self.predict_score(fst, snd, "pd") - \
            self.predict_score(fst, third, "pd")
        ranking_loss = -np.log(sigmoid(x))

        complexity = 0.0
        complexity += self.matrix_reg * np.dot(self.paper_latent_matrix[fst],
                                               self.paper_latent_matrix[fst])
        complexity += self.matrix_reg * np.dot(self.author_latent_matrix[snd],
                                               self.author_latent_matrix[snd])
        complexity += self.matrix_reg * np.dot(self.author_latent_matrix[third],
                                               self.author_latent_matrix[third])
        return ranking_loss + complexity

    def compute_dd_loss(self, fst, snd, third):
        x = self.predict_score(fst, snd, "dd") - \
            self.predict_score(fst, third, "dd")
        ranking_loss = -np.log(sigmoid(x))

        complexity = 0.0
        complexity += self.matrix_reg * np.dot(self.paper_latent_matrix[fst],
                                               self.paper_latent_matrix[fst])
        complexity += self.matrix_reg * np.dot(self.paper_latent_matrix[snd],
                                               self.paper_latent_matrix[snd])
        complexity += self.matrix_reg * np.dot(self.paper_latent_matrix[third],
                                               self.paper_latent_matrix[third])
        return ranking_loss + complexity

    def compute_dt_loss(self, fst, snd, third):
        x = self.predict_score(fst, snd, "dt") - \
            self.predict_score(fst, third, "dt")
        ranking_loss = -np.log(sigmoid(x))

        complexity = 0.0
        complexity += self.matrix_reg * np.dot(self.paper_latent_matrix[fst],
                                               self.paper_latent_matrix[fst])
        complexity += self.matrix_reg * np.dot(self.title_latent_matrix[snd],
                                               self.title_latent_matrix[snd])
        complexity += self.matrix_reg * np.dot(self.title_latent_matrix[third],
                                               self.title_latent_matrix[third])
        return ranking_loss + complexity

    def compute_djconf_loss(self, fst, snd, third):
        x = self.predict_score(fst, snd, "djconf") - \
            self.predict_score(fst, third, "djconf")
        ranking_loss = -np.log(sigmoid(x))

        complexity = 0.0
        complexity += self.matrix_reg * np.dot(self.paper_latent_matrix[fst],
                                               self.paper_latent_matrix[fst])
        complexity += self.matrix_reg * np.dot(self.jconf_latent_matrix[snd],
                                               self.jconf_latent_matrix[snd])
        complexity += self.matrix_reg * np.dot(self.jconf_latent_matrix[third],
                                               self.jconf_latent_matrix[third])
        return ranking_loss + complexity

    def compute_dorg_loss(self, fst, snd, third):
        x = self.predict_score(fst, snd, "dt") - \
            self.predict_score(fst, third, "dt")
        ranking_loss = -np.log(sigmoid(x))

        complexity = 0.0
        complexity += self.matrix_reg * np.dot(self.paper_latent_matrix[fst],
                                               self.paper_latent_matrix[fst])
        complexity += self.matrix_reg * np.dot(self.org_latent_matrix[snd],
                                               self.org_latent_matrix[snd])
        complexity += self.matrix_reg * np.dot(self.org_latent_matrix[third],
                                               self.org_latent_matrix[third])
        return ranking_loss + complexity

    def compute_dyear_loss(self, fst, snd, third):
        x = self.predict_score(fst, snd, "dyear") - \
            self.predict_score(fst, third, "dyear")
        ranking_loss = -np.log(sigmoid(x))

        complexity = 0.0
        complexity += self.matrix_reg * np.dot(self.paper_latent_matrix[fst],
                                               self.paper_latent_matrix[fst])
        complexity += self.matrix_reg * np.dot(self.year_latent_matrix[snd],
                                               self.year_latent_matrix[snd])
        complexity += self.matrix_reg * np.dot(self.year_latent_matrix[third],
                                               self.year_latent_matrix[third])
        return ranking_loss + complexity

    def compute_dabstract_loss(self, fst, snd, third):
        x = self.predict_score(fst, snd, "dabstract") - \
            self.predict_score(fst, third, "dabstract")
        ranking_loss = -np.log(sigmoid(x))

        complexity = 0.0
        complexity += self.matrix_reg * np.dot(self.paper_latent_matrix[fst],
                                               self.paper_latent_matrix[fst])
        complexity += self.matrix_reg * np.dot(self.abstract_latent_matrix[snd],
                                               self.abstract_latent_matrix[snd])
        complexity += self.matrix_reg * np.dot(self.abstract_latent_matrix[third],
                                               self.abstract_latent_matrix[third])
        return ranking_loss + complexity

    def predict_score(self, fst, snd, graph_type):
        """
        pp: person-person network
        pd: person-document bipartite network
        dd: doc-doc network
        detailed notation is inside paper
        """
        if graph_type == "pp":
            return np.dot(self.author_latent_matrix[fst],
                          self.author_latent_matrix[snd])
            # a = self.author_latent_matrix[fst]
            # b = self.author_latent_matrix[snd]
            # return np.dot(a,b)/float(np.sqrt(a.dot(a))*np.sqrt(b.dot(b)))
            # return np.sqrt(np.sum(np.square(self.author_latent_matrix[fst]
            #                                 - self.author_latent_matrix[snd])))
        elif graph_type == "pd":
            return np.dot(self.paper_latent_matrix[fst],
                          self.author_latent_matrix[snd])
            # a = self.paper_latent_matrix[fst]
            # b = self.author_latent_matrix[snd]
            # return np.dot(a, b) / float(np.sqrt(a.dot(a)) * np.sqrt(b.dot(b)))
            # return np.sqrt(np.sum(np.square(self.paper_latent_matrix[fst]
            #                                 - self.author_latent_matrix[snd])))
        elif graph_type == "dd":
            return np.dot(self.paper_latent_matrix[fst],
                          self.paper_latent_matrix[snd])
            # a = self.paper_latent_matrix[fst]
            # b = self.paper_latent_matrix[snd]
            # return np.dot(a, b) / float(np.sqrt(a.dot(a)) * np.sqrt(b.dot(b)))
            # return np.sqrt(np.sum(np.square(self.paper_latent_matrix[fst]
            #                                 - self.paper_latent_matrix[snd])))
        elif graph_type == "dt":
            return np.dot(self.paper_latent_matrix[fst],
                          self.title_latent_matrix[snd])
            # a = self.paper_latent_matrix[fst]
            # b = self.title_latent_matrix[snd]
            # return np.dot(a, b) / float(np.sqrt(a.dot(a)) * np.sqrt(b.dot(b)))
            # return np.sqrt(np.sum(np.square(self.paper_latent_matrix[fst]
            #                                 - self.title_latent_matrix[snd])))
        elif graph_type == "djconf":
            return np.dot(self.paper_latent_matrix[fst],
                          self.jconf_latent_matrix[snd])
            # a = self.paper_latent_matrix[fst]
            # b = self.jconf_latent_matrix[snd]
            # return np.dot(a, b) / float(np.sqrt(a.dot(a)) * np.sqrt(b.dot(b)))
            # return np.sqrt(np.sum(np.square(self.paper_latent_matrix[fst]
            #                                 - self.jconf_latent_matrix[snd])))
        elif graph_type == "dorg":
            return np.dot(self.paper_latent_matrix[fst],
                          self.org_latent_matrix[snd])
            # a = self.paper_latent_matrix[fst]
            # b = self.org_latent_matrix[snd]
            # return np.dot(a, b) / float(np.sqrt(a.dot(a)) * np.sqrt(b.dot(b)))
            # return np.sqrt(np.sum(np.square(self.paper_latent_matrix[fst]
            #                                 - self.org_latent_matrix[snd])))
        elif graph_type == "dyear":
            return np.dot(self.paper_latent_matrix[fst],
                          self.year_latent_matrix[snd])
            # a = self.paper_latent_matrix[fst]
            # b = self.year_latent_matrix[snd]
            # return np.dot(a, b) / float(np.sqrt(a.dot(a)) * np.sqrt(b.dot(b)))
            # return np.sqrt(np.sum(np.square(self.paper_latent_matrix[fst]
            #                                 - self.year_latent_matrix[snd])))
        elif graph_type == "dabstract":
            return np.dot(self.paper_latent_matrix[fst],
                          self.abstract_latent_matrix[snd])
            # a = self.paper_latent_matrix[fst]
            # b = self.abstract_latent_matrix[snd]
            # return np.dot(a, b) / float(np.sqrt(a.dot(a)) * np.sqrt(b.dot(b)))
            # return np.sqrt(np.sum(np.square(self.paper_latent_matrix[fst]
            #                                 - self.abstract_latent_matrix[snd])))