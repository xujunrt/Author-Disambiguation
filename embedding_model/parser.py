#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-12-11 下午7:09
# @Author  : xujun
# @Email   : xujunrt@163.com
# @Version : 1.0
import networkx as nx
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import re
import jieba.analyse


# 主要用于读取xml文件
class DataSet():

    def __init__(self, file_path):
        self.file_path = file_path
        self.paper_authorlist_dict = {}
        self.paper_list = []
        self.coauthor_list = []
        self.paper_title=[]
        self.paper_jconf=[]
        self.paper_org = []
        self.paper_year = []
        self.paper_abstract = []
        self.paper_jconf_dict={}
        self.paper_title_dict = {}
        self.paper_org_dict = {}
        self.paper_year_dict = {}
        self.paper_abstract_dict = {}
        # 如果label相同，那么一定是同一个人
        self.label_list = []
        self.C_Graph = nx.Graph()
        self.D_Graph = nx.Graph()
        self.T_Graph = nx.Graph()
        self.Jconf_Graph = nx.Graph()
        self.Org_Graph = nx.Graph()
        self.Year_Graph = nx.Graph()
        self.Abstract_Graph = nx.Graph()
        # 整个图中总的边数
        self.num_nnz = 0

    def reader_arnetminer(self):
        file_title_abstract = open('title_abstract.txt','r')
        title_abstract_dict = {}
        for line in file_title_abstract.readlines():
            if "<>" in line:
                arr=line.split("<>")
                if len(arr)>1:
                    title_abstract_dict[arr[0]]=arr[1]
                else:
                    title_abstract_dict[arr[0]]=""
        paper_index = 0
        coauthor_set = set()

        with open(self.file_path, "r") as filetoread:
            # 这里是一行一行读取的文件
            for line in filetoread:
                line = line.strip()
                if "FullName" in line:
                    ego_name = line[line.find('>')+1:line.rfind('<')].strip()
                elif "<title>" in line:
                    text = line[line.find('>')+1: line.rfind('<')].strip()
                    # 从这里找出对应的摘要
                    abstract = ""
                    self.paper_abstract.append(paper_index)
                    if text in title_abstract_dict:
                        abstract = title_abstract_dict.get(text)
                    if abstract=="":
                        abstract = text
                    abstract = re.sub("[:()'.,?`]", "",abstract)
                    abstract_list = []
                    if (len(abstract.split())>25):
                        abstract_list = jieba.analyse.extract_tags(abstract,topK=25)
                    else:
                        abstract_list = abstract.split()
                    self.paper_abstract_dict[paper_index] = []
                    for w in abstract_list:
                        try:
                            if w not in stopwords.words("english"):
                                # 将每个单词转换为原型
                                lemmatizer = WordNetLemmatizer()
                                yx = lemmatizer.lemmatize(w.lower())
                                self.paper_abstract_dict[paper_index].append(yx)
                        except:
                            continue


                    text = re.sub("[:()'.,?`]", "",text)
                    disease_List = text.split()
                    self.paper_title.append(paper_index)
                    self.paper_title_dict[paper_index]=[]
                    for w in disease_List:
                        try:
                            if w not in stopwords.words("english"):
                                # 将每个单词转换为原型
                                lemmatizer = WordNetLemmatizer()
                                yx = lemmatizer.lemmatize(w.lower())
                                self.paper_title_dict[paper_index].append(yx)
                        except:
                            continue

                elif "<jconf>" in line:
                    text = line[line.find('>') + 1: line.rfind('<')].strip()
                    text = re.sub("[:()'.,?`]", "", text)
                    disease_List = text.split()
                    self.paper_jconf.append(paper_index)
                    self.paper_jconf_dict[paper_index] = []
                    for w in disease_List:
                        if w not in stopwords.words("english"):
                            self.paper_jconf_dict[paper_index].append(w)
                        try:
                            if w not in stopwords.words("english"):
                                # 将每个单词转换为原型
                                lemmatizer = WordNetLemmatizer()
                                yx = lemmatizer.lemmatize(w.lower())
                                self.paper_jconf_dict[paper_index].append(yx)
                        except:
                            continue

                elif "<publication>" in line:
                    paper_index += 1
                    # 加入第几篇文章
                    self.paper_list.append(paper_index)
                elif "<year>" in line:
                    self.paper_year.append(paper_index)
                    self.paper_year_dict[paper_index] = []
                    text = line[line.find('>') + 1: line.rfind('<')].strip()
                    if text != "null":
                        self.paper_year_dict[paper_index].append(text)
                elif "<organization>" in line:
                    text = line[line.find('>') + 1: line.rfind('<')].strip()
                    text = re.sub("[:()'.,?`]", "", text)
                    disease_List = text.split()
                    self.paper_org.append(paper_index)
                    self.paper_org_dict[paper_index] = []
                    for w in disease_List:
                        try:
                            if w not in stopwords.words("english"):
                                # 将每个单词转换为原型
                                lemmatizer = WordNetLemmatizer()
                                yx = lemmatizer.lemmatize(w.lower())
                                self.paper_org_dict[paper_index].append(yx)
                        except:
                            continue

                elif "<authors>" in line:
                    # 加入的是所有的作者
                    author_list = line[line.find('>')+1: line.rfind('<')].strip().split(',')
                    if len(author_list) > 1:
                        if ego_name in author_list:
                            # 这里删除了文章的作者
                            author_list.remove(ego_name)
                            # 显示第几篇文章的合作者列表
                            self.paper_authorlist_dict[paper_index] = author_list
                        else:
                            self.paper_authorlist_dict[paper_index] = author_list

                        for co_author in author_list:
                            coauthor_set.add(co_author)

                        # 构建合作者图谱，只针对每一篇文章的作者集合中做
                        for pos in xrange(0, len(author_list) - 1):
                            for inpos in xrange(pos+1, len(author_list)):
                                # 得到的是一种单向合作关系
                                src_node = author_list[pos]
                                dest_node = author_list[inpos]
                                if not self.C_Graph.has_edge(src_node, dest_node):
                                    self.C_Graph.add_edge(src_node, dest_node, weight = 1)
                                else:
                                    edge_weight = self.C_Graph[src_node][dest_node]['weight']
                                    # 合作次数越多，加入的权值越大
                                    edge_weight += 1
                                    self.C_Graph[src_node][dest_node]['weight'] = edge_weight
                    else:
                        self.paper_authorlist_dict[paper_index] = []
                elif "<label>" in line:
                    label = int(line[line.find('>')+1: line.rfind('<')].strip())
                    self.label_list.append(label)

        self.coauthor_list = list(coauthor_set)
        """
        compute the 2-extension coauthorship for each paper
        generate doc-doc network
        edge weight is based on 2-coauthorship relation
        edge weight details are in paper definition 3.3
        """
        paper_2hop_dict = {}
        for paper_idx in self.paper_list:
            temp = set()
            if self.paper_authorlist_dict[paper_idx] != []:
                for first_hop in self.paper_authorlist_dict[paper_idx]:
                    temp.add(first_hop)
                    if self.C_Graph.has_node(first_hop):
                        for snd_hop in self.C_Graph.neighbors(first_hop):
                            temp.add(snd_hop)
            paper_2hop_dict[paper_idx] = temp
        for i in self.paper_title:
            for j in self.paper_title:
                if i != j:
                    title_set1 = self.paper_title_dict[i]
                    title_set2 = self.paper_title_dict[j]
                    title_edge_weight = len((set(title_set1)).intersection(set(title_set2)))
                    #print title_edge_weight,"weight"
                    if title_edge_weight != 0:
                        self.T_Graph.add_edge(i,j,weight=title_edge_weight)

        for i in self.paper_jconf:
            for j in self.paper_jconf:
                if i != j:
                    jconf1 = self.paper_title_dict[i]
                    jconf2 = self.paper_title_dict[j]
                    jconf_edge_weight = len((set(jconf1)).intersection(set(jconf1)))
                    #print title_edge_weight,"weight"
                    if jconf_edge_weight != 0:
                        self.Jconf_Graph.add_edge(i,j,weight=jconf_edge_weight)
        for i in self.paper_org:
            for j in self.paper_org:
                if i != j:
                    org1 = self.paper_org_dict[i]
                    org2 = self.paper_org_dict[j]
                    org_edge_weight = len((set(org1)).intersection(set(org2)))
                    #print title_edge_weight,"weight"
                    if org_edge_weight != 0:
                        self.Org_Graph.add_edge(i,j,weight=org_edge_weight)
        for i in self.paper_year:
            for j in self.paper_year:
                if i != j:
                    year1 = self.paper_year_dict[i]
                    year2 = self.paper_year_dict[j]
                    year_edge_weight = 0
                    if abs(int(year1[0])-int(year2[0]))<20:
                        year_edge_weight = 1
                    #print title_edge_weight,"weight"
                    if year_edge_weight != 0:
                        self.Year_Graph.add_edge(i,j,weight=year_edge_weight)

        for i in self.paper_abstract:
            for j in self.paper_abstract:
                if i != j:
                    abstract1 = self.paper_abstract_dict[i]
                    abstract2 = self.paper_abstract_dict[j]
                    abstract_edge_weight = len((set(abstract1)).intersection(set(abstract2)))
                    # print title_edge_weight,"weight"
                    if abstract_edge_weight != 0:
                        self.Abstract_Graph.add_edge(i, j, weight=abstract_edge_weight)

        for idx1 in xrange(0, len(self.paper_list) - 1):
            for idx2 in xrange(idx1 + 1, len(self.paper_list)):
                temp_set1 = paper_2hop_dict[self.paper_list[idx1]]
                temp_set2 = paper_2hop_dict[self.paper_list[idx2]]

                edge_weight = len(temp_set1.intersection(temp_set2))
                # print edge_weight, "weight"
                if edge_weight != 0:

                    self.D_Graph.add_edge(self.paper_list[idx1],
                                          self.paper_list[idx2],
                                          weight = edge_weight)
        bipartite_num_edge = 0
        for key, val in self.paper_authorlist_dict.items():
            if val != []:
                bipartite_num_edge += len(val)

        # self.num_nnz = self.D_Graph.number_of_edges() + \
        #                self.T_Graph.number_of_edges() + \
        #                self.Abstract_Graph.number_of_edges()+\
        #                self.Org_Graph.number_of_edges()+\
        #                self.C_Graph.number_of_edges()+\
        #                self.Jconf_Graph.number_of_edges()+\
        #                bipartite_num_edge
        #self.num_nnz = self.C_Graph.number_of_edges()
        #print self.num_nnz,"+++++++"
        self.num_nnz=300
