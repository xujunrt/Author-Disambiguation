#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-12-11 下午7:09
# @Author  : xujun
# @Email   : xujunrt@163.com
# @Version : 1.0
import parser
import embedding
import train_helper
import sampler
import eval_metric
import os
import time
import multiprocessing as mp
import pandas as pd
F1_List = []
F1_Max_List = []
F1_Max_List_pre =[]
F1_Max_List_rec = []
f1_name = {}


def get_file_list(file_dir):
	file_list = []
	for root, dirs, files in os.walk(file_dir):
		file_list.append(files)
	return file_list[0]

def get_median(data):
        data = sorted(data)
        size = len(data)
        if size % 2 == 0:   # 判断列表长度为偶数
            median = (data[size//2]+data[size//2-1])/2
            data[0] = median
        if size % 2 == 1:   # 判断列表长度为奇数
            median = data[(size-1)//2]
            data[0] = median
        return data[0]

def main(filename):
	"""
	pipeline for representation learning for all papers for a given name reference
	"""
	latent_dimen = 40
	alpha = 0.1
	matrix_reg = 0.1
	num_epoch = 2
	sampler_method = 'uniform'

	dataset = parser.DataSet(filename)
	dataset.reader_arnetminer()
	bpr_optimizer = embedding.BprOptimizer(latent_dimen, alpha, matrix_reg)
	# pp_sampler = sampler.CoauthorGraphSampler()
	# pd_sampler = sampler.BipartiteGraphSampler()
	dd_sampler = sampler.LinkedDocGraphSampler()
	dt_sampler = sampler.DocumentTitleSampler()
	djconf_sampler = sampler.DocumentJConfSampler()
	dyear_sampler = sampler.DocumentYearSampler()
	dorg_sampler = sampler.DocumentOrgSampler()
	dabstract_sampler = sampler.DocumentAbstractSampler()
	eval_f1 = eval_metric.Evaluator()

	run_helper = train_helper.TrainHelper()
	# avg_f1 = run_helper.helper(num_epoch, dataset, bpr_optimizer,
	#                            pp_sampler, pd_sampler, dd_sampler,
	#                            dt_sampler,djconf_sampler,
	#                            eval_f1, sampler_method)

	# 基本的，不加任何额外的东西
	avg_f1,avg_pre,avg_rec = run_helper.helper(num_epoch, dataset, bpr_optimizer,
	                           # pp_sampler,
	                           # pd_sampler,
	                           dd_sampler,
	                           dt_sampler,
	                           # djconf_sampler,
	                           # dorg_sampler,
	                           # dyear_sampler,
	                           dabstract_sampler,
	                           eval_f1, sampler_method,filename)

	F1_Max_List.append(avg_f1)
	F1_Max_List_pre.append(avg_pre)
	F1_Max_List_rec.append(avg_rec)

	print avg_f1


if __name__ == "__main__":
	file_list = get_file_list("../sampled_data/")
	file_list = sorted(file_list)
	file_list = file_list[:]
	cnt = 0
	copy_f1_list = []
	for x in file_list:
		cnt += 1
		filename = "../sampled_data/" + str(x)
		print filename
		print "count:" + str(cnt)
		print time.strftime('%H:%M:%S', time.localtime(time.time()))
		F1_Max_List = []
		for i in range(1):
			main(filename)
			print F1_Max_List
		f1_name[x]=[]
		f1_name[x].append(get_median(F1_Max_List))
		f1_name[x].append(max(F1_Max_List_pre))
		f1_name[x].append(max(F1_Max_List_rec))
		F1_List.append(max(F1_Max_List))
		copy_f1_list.append(max(F1_Max_List))
		print "real time f1:" + str(sum(F1_List) / len(F1_List))

	F1_List.sort()
	F1_List = F1_List[::-1]
	print F1_List
	print f1_name
	print sorted(f1_name.items(), lambda x, y: cmp(x[1], y[1]))
	print "top10 f1:" + str(sum(F1_List[0:10]) / 10)
	print "top30 f1:" + str(sum(F1_List[0:30]) / 30)
	print "top50 f1:" + str(sum(F1_List[0:50]) / 50)
	print "top70 f1:" + str(sum(F1_List[0:70]) / 70)
	print "top100 f1:" + str(sum(F1_List[0:100]) / 100)
	print "all f1:" + str(sum(F1_List) /7022)
	dataframe = pd.DataFrame({"author":file_list,"macro_f1":copy_f1_list})
	dataframe.to_csv("our_result.csv")
