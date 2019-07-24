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

results_path = '/home/space/Documents/vot-toolkit/VOT_workspace/MBMD_Hao2/results'
our_path = os.path.join(results_path, 'Ours', 'longterm')
vid_name = os.listdir(our_path)
vid_name.sort()


image_list = os.listdir('/home/space/Documents/Experiment/ICCV19/all_video_no_nan')
image_list.sort()
fps = 24   #视频帧率
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')




for vid_i in range(35):
    videoWriter = cv2.VideoWriter('/home/space/Documents/Experiment/ICCV19/avi_video/{:s}.avi'.format(vid_name[vid_i]), fourcc, fps, (640,360))
    
    for k in image_list:
        if k[:2] != '{:0>2d}'.format(vid_i):
            continue
        
        
        print 'processing...', k
        if k[3:5] == '00':
            for _ in range(50):
                img12 = cv2.imread('/home/space/Documents/Experiment/ICCV19/all_video_no_nan/' + k)
                img12 = cv2.resize(img12, (640,360))
                videoWriter.write(img12)
        img12 = cv2.imread('/home/space/Documents/Experiment/ICCV19/all_video_no_nan/' + k)
        videoWriter.write(img12)
    videoWriter.release()

    from subprocess import call
    dir1 = '/home/space/Documents/Experiment/ICCV19/avi_video/{:s}'.format(vid_name[vid_i])
    dir2 = '/home/space/Documents/Experiment/ICCV19/mp4_video/{:s}'.format(vid_name[vid_i])
    command = "ffmpeg -i %s.avi -b 2000k %s.mp4" % (dir1, dir2)
    call(command.split())
