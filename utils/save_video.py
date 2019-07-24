# coding=utf-8

"""
0  ballet
1  bicycle
11 cat1 X
13 dragon 200-1600
15 freestyle x
24 person19 1800-3500 x
30 rollerman 300-1000
32 tightrope x
34 yamaha  0-1000
"""


import os
import cv2

image_list = os.listdir('/home/space/Documents/Experiment/ICCV19/video')
image_list.sort()
fps = 24   #视频帧率
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
videoWriter = cv2.VideoWriter('/home/space/Documents/Experiment/ICCV19/save.avi', fourcc, fps, (640,360))


delete = ['15', '32', '24', '11','00']

long = []
for k in image_list:
    if k[:2] in delete:
        continue

#    if (k[:2] == '00' and k[3:5] == '01'
#        and  float(k[6:11]) not in range(0,1001)):
#            continue
    if (k[:2] == '01' and k[3:5] == '01'
        and  float(k[6:11]) not in range(0,1601)):
            continue
    if (k[:2] == '13' and k[3:5] == '01'
        and  float(k[6:11]) not in range(200,1601)):
            continue
    if (k[:2] == '24' and k[3:5] == '01'
        and  float(k[6:11]) not in range(1800,3501)):
            continue
#    if (k[:2] == '30' and k[3:5] == '01'
#        and  float(k[6:11]) not in range(300,1001)):
#            continue
    if (k[:2] == '34' and k[3:5] == '01'
        and  float(k[6:11]) not in range(0,1001)):
            continue
    
    
    print 'processing...', k
    if k[3:5] == '00':
        for _ in range(50):
            img12 = cv2.imread('/home/space/Documents/Experiment/ICCV19/video/' + k)
            img12 = cv2.resize(img12, (640,360))
            videoWriter.write(img12)
    img12 = cv2.imread('/home/space/Documents/Experiment/ICCV19/video/' + k)
    videoWriter.write(img12)
videoWriter.release()

from subprocess import call
dir = '/home/space/Documents/Experiment/ICCV19/save'
command = "ffmpeg -i %s.avi -b 2000k %s.mp4" % (dir, dir)
call(command.split())
