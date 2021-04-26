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
#Selected Classes and Grouping to Super Classes
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

base_dir =  '/home/anam/Desktop/Leipzig/Subjects_Data/AgeGroup_003/257498/Video/MPILab_0002_257498_01_P_Video'
coco_dir = os.path.join(base_dir,'Coco_Detections')
quantex_dir = os.path.join(base_dir,'QuantEx_Detections')
labels_dir = os.path.join(base_dir,'labels')

coco_files = coco_dir + '/*.txt'

for f in sorted(glob.glob(coco_files)):
        path,filename = os.path.split(f)
        filename,ext = os.path.splitext(filename)
        strings = filename.split('_')
        strings[-1] = strings[-1].zfill(4)
        new_file = 'frame_' + strings[-1]+'.txt'
        new_file = os.path.join(coco_dir,new_file)
        os.rename(f,new_file)

count = 0
for  src_file in  sorted(glob.glob(coco_files)):
    path,filename = os.path.split(src_file)
    filename,ext = os.path.splitext(filename)
    strings = filename.split('_')
    if int(strings[-1])%30==0:
        fr_num = int(int(strings[-1])/30)
        cnt = str(fr_num).zfill(4)
        new_file = 'frame_'+cnt+'.txt'
        new_file = os.path.join(quantex_dir,new_file)
        #count+=1
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

quantex_files = quantex_dir + '/*.txt'
tr = 'train.txt'
train_file = os.path.join(labels_dir,tr)
for  src_file in  sorted(glob.glob(quantex_files)):
    path,filename = os.path.split(src_file)
    data = 'obj_train_data/'+filename
    with open(train_file,"a") as fl:
        data.encode("utf8")
        fl.write(data)
        fl.write('\n')

