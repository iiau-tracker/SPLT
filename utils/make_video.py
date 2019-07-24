# coding=utf-8

import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw

results_path = '/home/space/Documents/vot-toolkit/VOT_workspace/MBMD_Hao2/results'
LTB_path = '/media/space/EAGET1T/VOT/LTB35'

our_path = os.path.join(results_path, 'Ours', 'longterm')
mbmd_path = os.path.join(results_path, 'MBMD', 'longterm')
DaSiam_path = os.path.join(results_path, 'DaSiam_LT', 'longterm')

def show_save(im, box, win_name='', frame_id=None, save=True):
    cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
    cv2.rectangle(im, (box[1], box[0]),
                  (box[3], box[2]), [0, 255, 0], 2)
    im = cv2.resize(im, (640,360))
    if frame_id is not None:
        cv2.putText(im,'#'+str(frame_id),(5,25), cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),2)
    if save:
        cv2.imwrite("/home/space/Documents/Experiment/ICCV19/video/01_%05d.jpg"%frame_id, im)
    cv2.imshow(win_name, im)
    cv2.waitKey(1)


#%%

vid_name = os.listdir(our_path)
vid_name.sort()

"""
0  ballet
1  bicycle
11 cat1
13 dragon 200-1600
15 freestyle x
24 person19 1800-3500
30 rollerman 300-1000
32 tightrope x
34 yamaha  0-1000
"""
for vid_id in [33]:#[0,1,11,15,24,32,34]:
    with open(os.path.join(our_path, vid_name[vid_id], vid_name[vid_id]+'_001.txt')) as f:
        bbox_list = f.readlines()[1:]
    
    bbox1 = []
    for line in bbox_list:
        tmp = line.split(',')
        bbox1.append(np.array([float(p) for p in tmp]))
    bbox1 = np.array(bbox1).astype(int)   
        
        
    with open(os.path.join(mbmd_path, vid_name[vid_id], vid_name[vid_id]+'_001.txt')) as f:
        bbox_list = f.readlines()[1:]
    
    bbox2 = []
    for line in bbox_list:
        tmp = line.split(',')
        bbox2.append(np.array([float(p) for p in tmp]))
    bbox2 = np.array(bbox2).astype(int)   
        
    
    
    with open(os.path.join(DaSiam_path, vid_name[vid_id], vid_name[vid_id]+'_001.txt')) as f:
        bbox_list = f.readlines()[1:]
    
    bbox3 = []
    for line in bbox_list:
        tmp = line.split(',')
        bbox3.append(np.array([float(p) for p in tmp]))
    bbox3 = np.array(bbox3).astype(int)   
    
    
    bbox1[:,2] += bbox1[:,0]
    bbox1[:,3] += bbox1[:,1]
    
    bbox2[:,2] += bbox2[:,0]
    bbox2[:,3] += bbox2[:,1]
    
    bbox3[:,2] += bbox3[:,0]
    bbox3[:,3] += bbox3[:,1]
    
    
    with open(os.path.join(LTB_path, vid_name[vid_id], 'groundtruth.txt')) as f:
        bbox_list = f.readlines()[1:]
    
    bbox = []
    for line in bbox_list:
        tmp = line.split(',')
        bbox.append(np.array([float(p) for p in tmp]))
    bbox = np.array(bbox)
    
    
            
    cv2.namedWindow('',cv2.WINDOW_NORMAL)
    for frame_i in range(0,len(bbox)):
        im_path = os.path.join(LTB_path, vid_name[vid_id], 'color', '{:0>8d}.jpg'.format(frame_i+1))
        im = Image.open(im_path)
        draw = ImageDraw.Draw(im)
    
#        if not np.isnan(bbox[frame_i][0]):
        box = bbox3[frame_i]
        draw.rectangle((box[0], box[1], box[2], box[3]), outline=(0,0,255), width=5)

        box = bbox2[frame_i]
        draw.rectangle((box[0], box[1], box[2], box[3]), outline=(255,0,0), width=5)

        box = bbox1[frame_i]
        draw.rectangle((box[0], box[1], box[2], box[3]), outline=(0,255,0), width=5)
    
            
        im = im.resize((640*2,360*2))
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype("arial.ttf", 30)
        draw.text((5,5), '#{:d}'.format(frame_i+1), fill = (255, 255 ,0), font=font)
        draw.text((5,330), 'Ours', fill = (0, 255 ,0), font=font)
        draw.text((95,330), 'MBMD', fill = (255, 0 ,0), font=font)
        draw.text((205,330), 'DaSiamRPN', fill = (0, 0 ,255), font=font)
        im.save("/home/space/Documents/Experiment/ICCV19/video/{:0>2d}_01_{:0>5d}.jpg".format(vid_id,frame_i+1))
        cv2.imshow('', np.array(im).astype('uint8')[:,:,::-1])
        cv2.waitKey(20)



