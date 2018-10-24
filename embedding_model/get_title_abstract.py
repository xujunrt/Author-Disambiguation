#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-1-6 上午9:29
# @Author  : xujun
# @Email   : xujunrt@163.com
# @Version : 1.0

with open('title_abstract.txt','r') as f:
	line_list =f.readlines()
	new_line_list = []
	str_item = []
	for item in line_list:
		if "<>" in item:
			str_item.append(item)
		else:
			str_item = str_item + item