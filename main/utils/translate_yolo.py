# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 12:41:59 2018

@author: crrodger
"""

import os

class_file = ''
labels_loc = ''
all_classes = []
img_height = 800
img_width = 944

def load_classes():
    f_class = open(class_file, 'r')
    all_classes.append(f_class.readlines())


all_labels = os.listdir(labels_loc)

for f_label in all_labels:
    if not 'classes' in f_label: #Ignore the classes.txt file
        new_name = f_label + '.org'
        os.rename(f_label, new_name)
        f_orig = open(new_name, 'r')
        f_new = open(f_label, 'w')
        
        lines = f_orig.readlines()
        
        for line in lines:
            obj = line.split(sep=" ")
            out_class = all_classes[obj[0]]
            out_x1 = obj[1]
            out_line = 