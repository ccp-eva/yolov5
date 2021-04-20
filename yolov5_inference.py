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

#model.classes = [0,72,73,74]
dst_path = '/home/anam/Codes/yolov5/Coco_Format_Detections'
quantex_path = '/home/anam/Codes/yolov5/QuantEx_Format_Detections'





##### YOLO O/P is in Format col-row (YX) ######

def label_locator(labels_list):
    im1 = cv2.imread('Frame_00000480.jpg')[:,:,::-1]
    test_result = model(im1)
    model_labels = test_result.names
    temp = set(labels_list)
    indexes = [i for i, val in enumerate(model_labels) if val in temp]
    return indexes

def video_detections(indexes):
    cap = cv2.VideoCapture('Test_Vid.mp4')
    count = 0
    #Inference 
    while(cap.isOpened()):
        ret,frame = cap.read()
        if ret:
            if count%30==0:
                results = model(frame)
                a = results.xywh[0].clone().detach()
                b =np.asanyarray( a.cpu().numpy())
                
                pix = results.xyxy[0].clone().detach()
                pix_np = np.asanyarray( pix.cpu().numpy())
                #print(pix_np)
                r,c = np.shape(b)
                if r:
                    filename = 'frame_' + str(count).zfill(4)+'.txt'
                    coco_file = os.path.join(dst_path,filename)
                    #np.savetxt(coco_file,b,fmt='%10.5f')#delimiter=' ', newline='\n')                    
                    for i in range(r):
                        super_cat = super_categ_finder(indexes,b[i,5])
                        b[i,5] = super_cat
                    quantex_b = np.delete(b,4,axis=1)
                    quantex_b = np.roll(quantex_b,1)
                    quantex_file = os.path.join(quantex_path,filename)
                    np.savetxt(quantex_file,quantex_b)

        elif ~ret:
            break
        count+=1
    cap.release()
    cv2.destroyAllWindows()
    print('Done')
    return
def super_categ_finder(indexes,value):
    for sub_list in indexes:
        if value in sub_list:
            label,loc = (indexes.index(sub_list), sub_list.index(value))
            break
        else:
            label = 6
    return label
    

def Face_Detector(roi):
    # Load the cascade
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    faces = face_cascade.detectMultiScale(image=roi, scaleFactor=1.3, minNeighbors=4)
    profiles = profile_cascade.detectMultiScale(image=roi, scaleFactor=1.3, minNeighbors=4)
    
    return faces,profiles


def Draw_face(face,image,color):
        for (x, y, width, height) in face:
            cv2.rectangle(
                image,
                (x, y),
                (x + width, y + height),
                color,
                thickness=2
            )

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


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
#indexes = [[person_ind],[screen_ind],[cuttlery_ind],[animal_ind],[book_ind],[toy_ind]]
indexes = [person_ind,animal_ind,book_ind,cuttlery_ind,screen_ind,toy_ind]
video_detections(indexes)
#test_path = '/home/anam/Desktop/Leipzig/Subjects_Data/'
#list_files(test_path)