#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-1-5 上午10:42
# @Author  : xujun
# @Email   : xujunrt@163.com
# @Version : 1.0
import os
import time
import urllib2
import urllib
import requests
import json

def get(url, headers,datas=None):
  response = requests.request("GET",url, params=datas,headers=headers)
  print response
  json = response.json()
  return json

def get_file_list(file_dir):
	file_list = []
	for root, dirs, files in os.walk(file_dir):
		file_list.append(files)
	return file_list[0]

if __name__ == '__main__':
	file_list = get_file_list("../sampled_data/")
	file_list = sorted(file_list)
	file_list = file_list[:]
	cnt = 0
	paper_title_list = []
	for x in file_list:
		filename = "../sampled_data/" + str(x)
		print filename
		print time.strftime('%H:%M:%S', time.localtime(time.time()))
		with open(filename, "r") as filetoread:
			# 这里是一行一行读取的文件
			for line in filetoread:
				line = line.strip()
				if "<title>" in line:
					paper_title = line[line.find('>') + 1:line.rfind('<')].strip()
					print paper_title
					paper_title_list.append(paper_title)
	paper_title_list = sorted(list(set(paper_title_list)))

	fil = open("title_list.txt","w")
	for item in paper_title_list:
		fil.write(item)
		fil.write("\n")

	paper_title_list = paper_title_list[845+1260:2500]

	fileobj = open("title_abstract.txt","a")
	url = "https://api.aminer.org/api/search/pub?query="
	m = 0
	for item in paper_title_list:
		cnt += 1
		print "count",str(cnt)
		newurl=url+"+".join(item.split())
		newurl+="&size=20&sort=relevance"
		headers = {'Content-Type': 'application/json',
		           'User-Agent': "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.108 Safari/537.36"}
		headers["Referer"]="https://aminer.org/search?t=b&q="+"%20".join(item.split())
		print headers["Referer"]
		print newurl
		res = get(newurl,headers)
		abstract="null"
		title_abstract = item + "<>" + abstract
		try:
			m+=1
			print "not null num",m
			abstract = res["result"][0]["abstract"]
			title_abstract = item+"<>"+abstract.decode("utf-8")
		except:
			continue
		fileobj.write(title_abstract)
		fileobj.write("\n")