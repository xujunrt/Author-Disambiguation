#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:07:41 2018

@author: xujun
"""
import csv
import os

def get_file_list(file_dir):
	file_list = []
	for root, dirs, files in os.walk(file_dir):
		file_list.append(files)
	return file_list[0]

if __name__ == "__main__":
   file_list = get_file_list("./backup/")
   file_list = sorted(file_list)
   file_list = file_list[:]
   result_csv = open("aminer_author.csv","w")
   writer = csv.writer(result_csv)
   for x in file_list:
       filename = "./backup/" + str(x)
       paper_number = 0
       author_name = str(x)
       paper_label_set = set()
       
       with open(filename,"r") as filetoread:
           for line in filetoread:
               if "<publication>" in line:
                   paper_number = paper_number+1
               if "<label>" in line:
                   text = line[line.find('>')+1: line.rfind('<')].strip()
                   paper_label_set.add(text)

       writer.writerow([author_name,paper_number,len(paper_label_set)])
        
                   
                   
       