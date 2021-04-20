import glob
import os
import sys
import numpy as np 
import cv2 
import torch
from PIL import Image
import pandas as pd 

#Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s',pretrained=True)
model = model.autoshape()


def label_locator(labels_list):
    im1 = cv2.imread('Frame_00000480.jpg')[:,:,::-1]
    test_result = model(im1)
    model_labels = test_result.names
    temp = set(labels_list)
    indexes = [i for i, val in enumerate(model_labels) if val in temp]
    return indexes
def super_categ_finder(indexes,value):
    for sub_list in indexes:
        if value in sub_list:
            label,loc = (indexes.index(sub_list), sub_list.index(value))
            break
        else:
            label = 6
    return label
############################################################################
person = ['person']
person_ind = label_locator(person)
animal = ['bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe']
animal_ind = label_locator(animal)
book = ['book']
book_ind = label_locator(book)
cuttlery = ['wine glass','cup','fork','knife','spoon','bowl','dining table']
cuttlery_ind = label_locator(cuttlery)
screen = ['tv','laptop','cell phone']
screen_ind = label_locator(screen)
toy  = ['frisbee','sports ball','kite','teddy bear']
toy_ind = label_locator(toy)
indexes = [person_ind,animal_ind,book_ind,cuttlery_ind,screen_ind,toy_ind]

####################################################################################
"""
dst_folder = '/home/anam/Codes/yolov5/Coco_Format_Detections'
src_dir = '/home/anam/Codes/yolov5/runs/detect/exp/labels'
src_files = '/home/anam/Codes/yolov5/runs/detect/exp/labels/*.txt'
dst_dir = '/home/anam/Codes/yolov5/QuantEx_Format_Detections'

#files = [i for i in os.listdir(src_dir)]

#This part of code is to move files from detections to Coco_Folder
for f in  sorted(glob.glob(src_files)):
        path,filename = os.path.split(f)
        filename,ext = os.path.splitext(filename)
        strings = filename.split('_')
        strings[-1] = strings[-1].zfill(4)
        if (int(strings[-1])%30==0):
            new_file = 'frame_' + strings[-1]+'.txt'
            new_file = os.path.join(dst_folder,new_file)
            os.rename(f,new_file)
"""
src_dir = '/home/anam/Codes/yolov5/Coco_Format_Detections'
src_files = '/home/anam/Codes/yolov5/Coco_Format_Detections/*.txt'
dst_dir = '/home/anam/Codes/yolov5/QuantEx_Format_Detections'

for src_file in  sorted(glob.glob(src_files)):
    path,filename = os.path.split(src_file)
    new_file = os.path.join(dst_dir,filename)
    with open(src_file) as f:
        with open(new_file,"a") as fl:
            for line in f:
                line_data = line.split()
                #print(line_data)
                categ = int(line_data[0])
                super_cat = super_categ_finder(indexes,categ)
                line_data[0] = str(super_cat)
                new_line = " ".join(line_data)
                new_line.encode("utf8")
                fl.write(new_line)
                fl.write('\n')
            
