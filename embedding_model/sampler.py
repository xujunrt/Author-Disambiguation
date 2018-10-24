#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-12-11 下午7:09
# @Author  : xujun
# @Email   : xujunrt@163.com
# @Version : 1.0
import numpy as np
import random
from utility import softmax


"""
(i, j) belongs positive sample set
(i, t) belongs negative sample set
notation details are in the paper
用于采样产生正负集合
"""

class CoauthorGraphSampler():
    @staticmethod
    def generate_triplet_uniform(dataset):
        """
        sample negative instance uniformly
        """
        a_i = random.choice(dataset.C_Graph.nodes())
        a_t = random.choice(dataset.coauthor_list)

        while True:
            neig_list = dataset.C_Graph.neighbors(a_i)
            if a_t not in neig_list:
                # given a_i, sample its neighbor based on its weight value
                # idea of edge sampling
                weight_list = [dataset.C_Graph[a_i][nbr]['weight']
                               for nbr in neig_list]
                norm_weight_list = [float(w) / sum(weight_list)
                                    for w in weight_list]
                a_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]
                yield a_i, a_j, a_t
                break

            else:
                a_i = random.choice(dataset.C_Graph.nodes())
                a_t = random.choice(dataset.coauthor_list)

    @staticmethod
    def generate_triplet_reject(dataset, bpr_optimizer):
        """
        generate negative instance using ranking-aware rejection sampler
        consider linear case
        """
        a_i = random.choice(dataset.C_Graph.nodes())
        neg_pair = random.sample(dataset.coauthor_list, 2)

        while True:
            neig_list = dataset.C_Graph.neighbors(a_i)
            if neg_pair[0] not in neig_list and neg_pair[1] not in neig_list:
                # given a_i, sample its neighbor based on its weight value
                # idea of edge sampling
                weight_list = [dataset.C_Graph[a_i][nbr]['weight']
                               for nbr in neig_list]
                norm_weight_list = [float(w) / sum(weight_list)
                                    for w in weight_list]
                a_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]

                # sample negative instance using ranking-aware rejection sampler
                sc1 = bpr_optimizer.predict_score(a_i, neg_pair[0], "pp")
                sc2 = bpr_optimizer.predict_score(a_i, neg_pair[1], "pp")
                a_t = neg_pair[0] if sc1 <= sc2 else neg_pair[1]
                yield a_i, a_j, a_t
                break

            else:
                a_i = random.choice(dataset.C_Graph.nodes())
                neg_pair = random.sample(dataset.coauthor_list, 2)

    @staticmethod
    def generate_triplet_adaptive(dataset, bpr_optimizer):
        """
        generate negative instance using adaptive sampling
        sample from a pre-defined exponential distribution
        """
        a_i = random.choice(dataset.C_Graph.nodes())
        neg_list = list(set(dataset.coauthor_list) - \
                   set(dataset.C_Graph.neighbors(a_i)) - set([a_i]))

        # given a_i, sample its neighbor based on its weight value
        # idea of edge sampling
        neig_list = dataset.C_Graph.neighbors(a_i)
        weight_list = [dataset.C_Graph[a_i][nbr]['weight']
                       for nbr in neig_list]
        norm_weight_list = [float(w) / sum(weight_list)
                            for w in weight_list]
        a_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]

        # sample negative instance based on pre-defined exponential distribution
        norm_soft = softmax([bpr_optimizer.predict_score(a_i, ne, "pp")
                             for ne in neg_list])
        a_t = np.random.choice(neg_list, 1, p = norm_soft)[0]
        yield a_i, a_j, a_t


class LinkedDocGraphSampler():
    @staticmethod
    def generate_triplet_uniform(dataset):
        d_i = random.choice(dataset.D_Graph.nodes())
        d_t = random.choice(dataset.paper_list)

        while True:
            neig_list = dataset.D_Graph.neighbors(d_i)
            if d_t not in neig_list:
                # given d_i, sample its neighbor based on its weight value
                # idea of edge sampling
                weight_list = [dataset.D_Graph[d_i][nbr]['weight']
                               for nbr in neig_list]
                norm_weight_list = [float(w) / sum(weight_list)
                                    for w in weight_list]
                d_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]
                yield d_i, d_j, d_t
                break

            else:
                d_i = random.choice(dataset.D_Graph.nodes())
                d_t = random.choice(dataset.paper_list)

    @staticmethod
    def generate_triplet_reject(dataset, bpr_optimizer):
        """
        generate negative instance using ranking-aware rejection sampler
        consider linear case
        """
        d_i = random.choice(dataset.D_Graph.nodes())
        neg_pair = random.sample(dataset.paper_list, 2)

        while True:
            neig_list = dataset.D_Graph.neighbors(d_i)
            if neg_pair[0] not in neig_list and neg_pair[1] not in neig_list:
                # given a_i, sample its neighbor based on its weight value
                # idea of edge sampling
                weight_list = [dataset.D_Graph[d_i][nbr]['weight']
                               for nbr in neig_list]
                norm_weight_list = [float(w) / sum(weight_list)
                                    for w in weight_list]
                d_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]

                # sample negative instance using ranking-aware rejection sampler
                sc1 = bpr_optimizer.predict_score(d_i, neg_pair[0], "dd")
                sc2 = bpr_optimizer.predict_score(d_i, neg_pair[1], "dd")
                d_t = neg_pair[0] if sc1 <= sc2 else neg_pair[1]
                yield d_i, d_j, d_t
                break

            else:
                d_i = random.choice(dataset.D_Graph.nodes())
                neg_pair = random.sample(dataset.paper_list, 2)

    @staticmethod
    def generate_triplet_adaptive(dataset, bpr_optimizer):
        """
        generate negative instance using adaptive sampling
        sample from a pre-defined exponential distribution
        """
        d_i = random.choice(dataset.D_Graph.nodes())
        neg_list = list(set(dataset.paper_list) - \
                   set(dataset.D_Graph.neighbors(d_i)) - set([d_i]))

        # given a_i, sample its neighbor based on its weight value
        # idea of edge sampling
        neig_list = dataset.D_Graph.neighbors(d_i)
        weight_list = [dataset.D_Graph[d_i][nbr]['weight']
                       for nbr in neig_list]
        norm_weight_list = [float(w) / sum(weight_list)
                            for w in weight_list]
        d_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]

        # sample negative instance based on pre-defined exponential distribution
        norm_soft = softmax([bpr_optimizer.predict_score(d_i, ne, "dd")
                             for ne in neg_list])
        d_t = np.random.choice(neg_list, 1, p = norm_soft)[0]
        yield d_i, d_j, d_t

class DocumentTitleSampler():
    @staticmethod
    def generate_triplet_uniform(dataset):
        t_i = random.choice(dataset.T_Graph.nodes())
        t_t = random.choice(dataset.paper_title)

        while True:
            neig_list = dataset.T_Graph.neighbors(t_i)
            if t_t not in neig_list:
                # given d_i, sample its neighbor based on its weight value
                # idea of edge sampling
                weight_list = [dataset.T_Graph[t_i][nbr]['weight']
                               for nbr in neig_list]
                norm_weight_list = [float(w) / sum(weight_list)
                                    for w in weight_list]
                t_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]
                yield t_i, t_j, t_t
                break

            else:
                t_i = random.choice(dataset.T_Graph.nodes())
                t_t = random.choice(dataset.paper_title)

    @staticmethod
    def generate_triplet_reject(dataset, bpr_optimizer):
        """
        generate negative instance using ranking-aware rejection sampler
        consider linear case
        """
        t_i = random.choice(dataset.T_Graph.nodes())
        neg_pair = random.sample(dataset.paper_title, 2)

        while True:
            neig_list = dataset.T_Graph.neighbors(t_i)
            if neg_pair[0] not in neig_list and neg_pair[1] not in neig_list:
                # given a_i, sample its neighbor based on its weight value
                # idea of edge sampling
                weight_list = [dataset.T_Graph[t_i][nbr]['weight']
                               for nbr in neig_list]
                norm_weight_list = [float(w) / sum(weight_list)
                                    for w in weight_list]
                t_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]

                # sample negative instance using ranking-aware rejection sampler
                sc1 = bpr_optimizer.predict_score(t_i, neg_pair[0], "dt")
                sc2 = bpr_optimizer.predict_score(t_i, neg_pair[1], "dt")
                t_t = neg_pair[0] if sc1 <= sc2 else neg_pair[1]
                yield t_i, t_j, t_t
                break

            else:
                t_i = random.choice(dataset.T_Graph.nodes())
                neg_pair = random.sample(dataset.paper_title, 2)

    @staticmethod
    def generate_triplet_adaptive(dataset, bpr_optimizer):
        """
        generate negative instance using adaptive sampling
        sample from a pre-defined exponential distribution
        """
        t_i = random.choice(dataset.T_Graph.nodes())
        neg_list = list(set(dataset.paper_title) - \
                   set(dataset.T_Graph.neighbors(t_i)) - set([t_i]))

        # given a_i, sample its neighbor based on its weight value
        # idea of edge sampling
        neig_list = dataset.T_Graph.neighbors(t_i)
        weight_list = [dataset.T_Graph[t_i][nbr]['weight']
                       for nbr in neig_list]
        norm_weight_list = [float(w) / sum(weight_list)
                            for w in weight_list]
        t_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]

        # sample negative instance based on pre-defined exponential distribution
        norm_soft = softmax([bpr_optimizer.predict_score(t_i, ne, "dt")
                             for ne in neg_list])
        t_t = np.random.choice(neg_list, 1, p = norm_soft)[0]
        yield t_i, t_j, t_t



class DocumentJConfSampler():
    @staticmethod
    def generate_triplet_uniform(dataset):
        jconf_i = random.choice(dataset.Jconf_Graph.nodes())
        jconf_t = random.choice(dataset.paper_jconf)

        while True:
            neig_list = dataset.Jconf_Graph.neighbors(jconf_i)
            if jconf_t not in neig_list:
                weight_list = [dataset.Jconf_Graph[jconf_i][nbr]['weight']
                               for nbr in neig_list]
                norm_weight_list = [float(w) / sum(weight_list)
                                    for w in weight_list]
                jconf_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]
                yield jconf_i, jconf_j, jconf_t
                break

            else:
                jconf_i = random.choice(dataset.Jconf_Graph.nodes())
                jconf_t = random.choice(dataset.paper_jconf)

    @staticmethod
    def generate_triplet_reject(dataset, bpr_optimizer):
        """
        generate negative instance using ranking-aware rejection sampler
        consider linear case
        """
        jconf_i = random.choice(dataset.Jconf_Graph.nodes())
        neg_pair = random.sample(dataset.paper_jconf, 2)
        cnt=0
        while True:
            neig_list = dataset.Jconf_Graph.neighbors(jconf_i)
            if neg_pair[0] not in neig_list and neg_pair[1] not in neig_list:
                # given a_i, sample its neighbor based on its weight value
                # idea of edge sampling
                weight_list = [dataset.Jconf_Graph[jconf_i][nbr]['weight']
                               for nbr in neig_list]
                norm_weight_list = [float(w) / sum(weight_list)
                                    for w in weight_list]
                jconf_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]

                # sample negative instance using ranking-aware rejection sampler
                sc1 = bpr_optimizer.predict_score(jconf_i, neg_pair[0], "djconf")
                sc2 = bpr_optimizer.predict_score(jconf_i, neg_pair[1], "djconf")
                jconf_t = neg_pair[0] if sc1 <= sc2 else neg_pair[1]
                yield jconf_i, jconf_j, jconf_t
                break

            else:
                cnt += 1
                if (cnt > 5):
                    neig_list = dataset.Jconf_Graph.neighbors(jconf_i)
                    weight_list = [dataset.Jconf_Graph[jconf_i][nbr]['weight']
                                   for nbr in neig_list]
                    norm_weight_list = [float(w) / sum(weight_list)
                                        for w in weight_list]
                    jconf_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]
                    jconf_t = random.choice(dataset.Jconf_Graph.nodes())
                    yield jconf_i, jconf_j, jconf_t
                    break

                jconf_i = random.choice(dataset.Jconf_Graph.nodes())
                neg_pair = random.sample(dataset.paper_jconf, 2)

    @staticmethod
    def generate_triplet_adaptive(dataset, bpr_optimizer):
        """
        generate negative instance using adaptive sampling
        sample from a pre-defined exponential distribution
        """
        jconf_i = random.choice(dataset.Jconf_Graph.nodes())
        neg_list = list(set(dataset.paper_jconf) - \
                   set(dataset.Jconf_Graph.neighbors(jconf_i)) - set([jconf_i]))

        # given a_i, sample its neighbor based on its weight value
        # idea of edge sampling
        neig_list = dataset.Jconf_Graph.neighbors(jconf_i)
        weight_list = [dataset.Jconf_Graph[jconf_i][nbr]['weight']
                       for nbr in neig_list]
        norm_weight_list = [float(w) / sum(weight_list)
                            for w in weight_list]
        jconf_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]

        # sample negative instance based on pre-defined exponential distribution
        jconf_t = 0
        if len(neg_list) > 0:
            norm_soft = softmax([bpr_optimizer.predict_score(jconf_i, ne, "djconf")
                                 for ne in neg_list])
            jconf_t = np.random.choice(neg_list, 1, p=norm_soft)[0]
        else:
            jconf_t = np.random.choice(neig_list)
        yield jconf_i, jconf_j, jconf_t


class BipartiteGraphSampler():
    @staticmethod
    def generate_triplet_uniform(dataset):
        d_i = random.choice(dataset.paper_list)
        a_t = random.choice(dataset.coauthor_list)

        while True:
            if dataset.paper_authorlist_dict[d_i] != [] \
                and a_t not in dataset.paper_authorlist_dict[d_i]:
                a_j = random.choice(dataset.paper_authorlist_dict[d_i])
                yield d_i, a_j, a_t
                break

            else:
                d_i = random.choice(dataset.paper_list)
                a_t = random.choice(dataset.coauthor_list)

    @staticmethod
    def generate_triplet_reject(dataset, bpr_optimizer):
        """
        generate negative instance using ranking-aware rejection sampler
        consider linear case
        """
        d_i = random.choice(dataset.paper_list)
        neg_pair = random.sample(dataset.coauthor_list, 2)

        while True:
            if dataset.paper_authorlist_dict[d_i] != [] \
                and neg_pair[0] not in dataset.paper_authorlist_dict[d_i] \
                    and neg_pair[1] not in dataset.paper_authorlist_dict[d_i]:

                a_j = random.choice(dataset.paper_authorlist_dict[d_i])

                # sample negative instance using ranking-aware rejection sampler
                sc1 = bpr_optimizer.predict_score(d_i, neg_pair[0], "pd")
                sc2 = bpr_optimizer.predict_score(d_i, neg_pair[1], "pd")
                a_t = neg_pair[0] if sc1 <= sc2 else neg_pair[1]
                yield d_i, a_j, a_t
                break

            else:
                d_i = random.choice(dataset.paper_list)
                neg_pair = random.sample(dataset.coauthor_list, 2)

    @staticmethod
    def generate_triplet_adaptive(dataset, bpr_optimizer):
        """
        generate negative instance using adaptive sampling
        sample from a pre-defined exponential distribution
        """
        d_i = random.choice(dataset.paper_list)
        neg_list = list(set(dataset.coauthor_list) - \
                        set(dataset.paper_authorlist_dict[d_i]))

        while True:
            if dataset.paper_authorlist_dict[d_i] != []:
                a_j = random.choice(dataset.paper_authorlist_dict[d_i])

                # sample negative instance based on pre-defined exponential distribution
                norm_soft = softmax([bpr_optimizer.predict_score(d_i, ne, "pd")
                                     for ne in neg_list])
                a_t = np.random.choice(neg_list, 1, p = norm_soft)[0]
                yield d_i, a_j, a_t
                break

            else:
                d_i = random.choice(dataset.paper_list)
                neg_list = list(set(dataset.coauthor_list) - \
                                set(dataset.paper_authorlist_dict[d_i]))

class DocumentOrgSampler():
    @staticmethod
    def generate_triplet_uniform(dataset):
        t_i = random.choice(dataset.Org_Graph.nodes())
        t_t = random.choice(dataset.paper_org)

        while True:
            neig_list = dataset.Org_Graph.neighbors(t_i)
            if t_t not in neig_list:
                # given d_i, sample its neighbor based on its weight value
                # idea of edge sampling
                weight_list = [dataset.Org_Graph[t_i][nbr]['weight']
                               for nbr in neig_list]
                norm_weight_list = [float(w) / sum(weight_list)
                                    for w in weight_list]
                t_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]
                yield t_i, t_j, t_t
                break

            else:
                t_i = random.choice(dataset.Org_Graph.nodes())
                t_t = random.choice(dataset.paper_org)

    @staticmethod
    def generate_triplet_reject(dataset, bpr_optimizer):
        """
        generate negative instance using ranking-aware rejection sampler
        consider linear case
        """
        t_i = random.choice(dataset.Org_Graph.nodes())
        neg_pair = random.sample(dataset.paper_org, 2)
        cnt = 0
        while True:
            neig_list = dataset.Org_Graph.neighbors(t_i)
            if neg_pair[0] not in neig_list and neg_pair[1] not in neig_list:
                # given a_i, sample its neighbor based on its weight value
                # idea of edge sampling
                weight_list = [dataset.Org_Graph[t_i][nbr]['weight']
                               for nbr in neig_list]
                norm_weight_list = [float(w) / sum(weight_list)
                                    for w in weight_list]
                t_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]

                # sample negative instance using ranking-aware rejection sampler
                sc1 = bpr_optimizer.predict_score(t_i, neg_pair[0], "dorg")
                sc2 = bpr_optimizer.predict_score(t_i, neg_pair[1], "dorg")
                t_t = neg_pair[0] if sc1 <= sc2 else neg_pair[1]
                yield t_i, t_j, t_t
                break

            else:
                cnt += 1
                if (cnt > 5):
                    neig_list = dataset.Org_Graph.neighbors(t_i)
                    weight_list = [dataset.Org_Graph[t_i][nbr]['weight']
                                   for nbr in neig_list]
                    norm_weight_list = [float(w) / sum(weight_list)
                                        for w in weight_list]
                    t_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]
                    t_t = random.choice(dataset.Org_Graph.nodes())
                    yield t_i, t_j, t_t
                    break
                t_i = random.choice(dataset.Org_Graph.nodes())
                neg_pair = random.sample(dataset.paper_org, 2)

    @staticmethod
    def generate_triplet_adaptive(dataset, bpr_optimizer):
        """
        generate negative instance using adaptive sampling
        sample from a pre-defined exponential distribution
        """
        t_i = random.choice(dataset.Org_Graph.nodes())
        neg_list = list(set(dataset.paper_org) - \
                   set(dataset.Org_Graph.neighbors(t_i)) - set([t_i]))

        # given a_i, sample its neighbor based on its weight value
        # idea of edge sampling
        neig_list = dataset.Org_Graph.neighbors(t_i)
        weight_list = [dataset.Org_Graph[t_i][nbr]['weight']
                       for nbr in neig_list]
        norm_weight_list = [float(w) / sum(weight_list)
                            for w in weight_list]
        t_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]

        # sample negative instance based on pre-defined exponential distribution
        t_t = 0
        if len(neg_list) > 0:
            norm_soft = softmax([bpr_optimizer.predict_score(t_i, ne, "dorg")
                                 for ne in neg_list])
            t_t = np.random.choice(neg_list, 1, p=norm_soft)[0]
        else:
            t_t = np.random.choice(neig_list)
        yield t_i, t_j, t_t

class DocumentYearSampler():
    @staticmethod
    def generate_triplet_uniform(dataset):
        t_i = random.choice(dataset.Year_Graph.nodes())
        t_t = random.choice(dataset.paper_year)

        while True:
            neig_list = dataset.Year_Graph.neighbors(t_i)
            if t_t not in neig_list:
                # given d_i, sample its neighbor based on its weight value
                # idea of edge sampling
                weight_list = [dataset.Year_Graph[t_i][nbr]['weight']
                               for nbr in neig_list]
                norm_weight_list = [float(w) / sum(weight_list)
                                    for w in weight_list]
                t_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]
                yield t_i, t_j, t_t
                break

            else:
                t_i = random.choice(dataset.Year_Graph.nodes())
                t_t = random.choice(dataset.paper_year)

    @staticmethod
    def generate_triplet_reject(dataset, bpr_optimizer):
        """
        generate negative instance using ranking-aware rejection sampler
        consider linear case
        """
        t_i = random.choice(dataset.Year_Graph.nodes())
        neg_pair = random.sample(dataset.paper_year, 2)

        while True:
            neig_list = dataset.Year_Graph.neighbors(t_i)
            if neg_pair[0] not in neig_list and neg_pair[1] not in neig_list:
                # given a_i, sample its neighbor based on its weight value
                # idea of edge sampling
                weight_list = [dataset.Year_Graph[t_i][nbr]['weight']
                               for nbr in neig_list]
                norm_weight_list = [float(w) / sum(weight_list)
                                    for w in weight_list]
                t_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]

                # sample negative instance using ranking-aware rejection sampler
                sc1 = bpr_optimizer.predict_score(t_i, neg_pair[0], "dyear")
                sc2 = bpr_optimizer.predict_score(t_i, neg_pair[1], "dyear")
                t_t = neg_pair[0] if sc1 <= sc2 else neg_pair[1]
                yield t_i, t_j, t_t
                break

            else:
                t_i = random.choice(dataset.Year_Graph.nodes())
                neg_pair = random.sample(dataset.paper_year, 2)

    @staticmethod
    def generate_triplet_adaptive(dataset, bpr_optimizer):
        """
        generate negative instance using adaptive sampling
        sample from a pre-defined exponential distribution
        """
        t_i = random.choice(dataset.Year_Graph.nodes())
        neg_list = list(set(dataset.paper_year) - \
                   set(dataset.Year_Graph.neighbors(t_i)) - set([t_i]))

        # given a_i, sample its neighbor based on its weight value
        # idea of edge sampling
        neig_list = dataset.Year_Graph.neighbors(t_i)
        weight_list = [dataset.Year_Graph[t_i][nbr]['weight']
                       for nbr in neig_list]
        norm_weight_list = [float(w) / sum(weight_list)
                            for w in weight_list]
        t_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]

        # sample negative instance based on pre-defined exponential distribution
        norm_soft = softmax([bpr_optimizer.predict_score(t_i, ne, "dyear")
                             for ne in neg_list])
        t_t = np.random.choice(neg_list, 1, p = norm_soft)[0]
        yield t_i, t_j, t_t

class DocumentAbstractSampler():
    @staticmethod
    def generate_triplet_uniform(dataset):
        t_i = random.choice(dataset.Abstract_Graph.nodes())
        t_t = random.choice(dataset.paper_abstract)

        while True:
            neig_list = dataset.Abstract_Graph.neighbors(t_i)
            if t_t not in neig_list:
                # given d_i, sample its neighbor based on its weight value
                # idea of edge sampling
                weight_list = [dataset.Abstract_Graph[t_i][nbr]['weight']
                               for nbr in neig_list]
                norm_weight_list = [float(w) / sum(weight_list)
                                    for w in weight_list]
                t_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]
                yield t_i, t_j, t_t
                break

            else:
                t_i = random.choice(dataset.Abstract_Graph.nodes())
                t_t = random.choice(dataset.paper_abstract)

    @staticmethod
    def generate_triplet_reject(dataset, bpr_optimizer):
        """
        generate negative instance using ranking-aware rejection sampler
        consider linear case
        """
        t_i = random.choice(dataset.Abstract_Graph.nodes())
        neg_pair = random.sample(dataset.paper_abstract, 2)
        cnt = 0
        while True:
            neig_list = dataset.Abstract_Graph.neighbors(t_i)
            if neg_pair[0] not in neig_list and neg_pair[1] not in neig_list:
                # given a_i, sample its neighbor based on its weight value
                # idea of edge sampling
                weight_list = [dataset.Abstract_Graph[t_i][nbr]['weight']
                               for nbr in neig_list]
                norm_weight_list = [float(w) / sum(weight_list)
                                    for w in weight_list]
                t_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]

                # sample negative instance using ranking-aware rejection sampler
                sc1 = bpr_optimizer.predict_score(t_i, neg_pair[0], "dabstract")
                sc2 = bpr_optimizer.predict_score(t_i, neg_pair[1], "dabstract")
                t_t = neg_pair[0] if sc1 <= sc2 else neg_pair[1]
                yield t_i, t_j, t_t
                break
            else:
                cnt+=1
                if(cnt>5):
                    neig_list = dataset.Abstract_Graph.neighbors(t_i)
                    weight_list = [dataset.Abstract_Graph[t_i][nbr]['weight']
                                   for nbr in neig_list]
                    norm_weight_list = [float(w) / sum(weight_list)
                                        for w in weight_list]
                    t_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]
                    t_t = random.choice(dataset.Abstract_Graph.nodes())
                    yield t_i, t_j, t_t
                    break
                t_i = random.choice(dataset.Abstract_Graph.nodes())
                neg_pair = random.sample(dataset.paper_abstract, 2)

    @staticmethod
    def generate_triplet_adaptive(dataset, bpr_optimizer):
        """
        generate negative instance using adaptive sampling
        sample from a pre-defined exponential distribution
        """
        t_i = random.choice(dataset.Abstract_Graph.nodes())
        neg_list = list(set(dataset.paper_abstract) - \
                        set(dataset.Abstract_Graph.neighbors(t_i)) - set([t_i]))

        # given a_i, sample its neighbor based on its weight value
        # idea of edge sampling
        neig_list = dataset.Abstract_Graph.neighbors(t_i)
        weight_list = [dataset.Abstract_Graph[t_i][nbr]['weight']
                       for nbr in neig_list]
        norm_weight_list = [float(w) / sum(weight_list)
                            for w in weight_list]
        t_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]

        # sample negative instance based on pre-defined exponential distribution
        t_t = 0
        if len(neg_list) > 0:
            norm_soft = softmax([bpr_optimizer.predict_score(t_i, ne, "dabstract")
                                 for ne in neg_list])
            t_t = np.random.choice(neg_list, 1, p=norm_soft)[0]
        else:
            t_t = np.random.choice(neig_list)
        yield t_i, t_j, t_t