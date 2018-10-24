#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-12-23 上午10:00
# @Author  : xujun
# @Email   : xujunrt@163.com
# @Version : 1.0

import os
import time

def get_file_list(file_dir):
	file_list = []
	for root, dirs, files in os.walk(file_dir):
		file_list.append(files)
	return file_list[0]

file_list = get_file_list("../sampled_data/")
file_list = sorted(file_list)
file_list = file_list[:]
cnt = 0
f = open("../feature.txt","wb")
for x in file_list:
	cnt += 1
	filename = "../sampled_data/" + str(x)
	print filename
	print "count:" + str(cnt)
	print time.strftime('%H:%M:%S', time.localtime(time.time()))
	with open(filename, "r") as filetoread:
		paper_dict = {}
		# 这里是一行一行读取的文件
		for line in filetoread:
			line = line.strip()
			if "<FullName>" in line:
				ego_name = line[line.find('>') + 1:line.rfind('<')].strip()
				paper_dict[ego_name]=[]
			elif "<title>" in line:
				paper_item = []
				text = line[line.find('>') + 1: line.rfind('<')].strip()
				paper_item.append(text)
			elif "<year>" in line:
				year = line[line.find('>') + 1: line.rfind('<')].strip()
				paper_item.append(year)
			elif "<authors>" in line:
				text = line[line.find('>') + 1: line.rfind('<')].strip()
				paper_item.append(text)
			elif "<jconf>" in line:
				text = line[line.find('>') + 1: line.rfind('<')].strip()
				paper_item.append(text)

			elif "<label>" in line:
				label = line[line.find('>') + 1: line.rfind('<')].strip()
				paper_item.append(label)
			elif "<organization>" in line:
				text = line[line.find('>') + 1: line.rfind('<')].strip()
				paper_item.append(text)
				paper_dict[ego_name].append(paper_item)
		print paper_dict
		for key,value in paper_dict.items():
			for item in value:
				line_list = []
				line_list.append(key)
				line_list+=item
				f.write("<>".join(line_list))
				f.write("\n")
				print len(line_list)
