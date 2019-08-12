#coding=utf-8
import numpy as np
import os
import cv2
from xml.etree import ElementTree as ET
import tensorflow as tf

'''generate triplet pairs and save in tfrecord for trainning :)'''
# The following path should be changed to your real path.
# Theoretically, TFRECORD_PATH could be anywhere, but SSD is recommended for higher speed. :)
file_dir = '/home/masterbin-iiau/Object_Tracking/project_for_OT/MBMD_Bin/VID/type_video'
VID_ROOT_DIR = '/media/masterbin-iiau/417507b2-6b1b-4e11-b488-4e0ff0e89a481/tangjiuqi097/ILSVRC2015_VID'
TFRECORD_PATH = '/media/masterbin-iiau/417507b2-6b1b-4e11-b488-4e0ff0e89a481/tangjiuqi097/VID_tfrecord/'


np.random.seed(1)
EPOCH = 40
PAIRS_PER_EPOCH = 60000
N = PAIRS_PER_EPOCH * EPOCH


aug_N = int(N * 1.04)
result = np.random.randint(0,30,(2,aug_N)).astype(np.uint8)
mask = (result[0] != result[1])
final_result = result[:,mask][:,:N]

print(final_result.shape)

type_names = ['airplane','antelope','bear','bicycle','bird',
              'bus','car','cattle','dog','domestic_cat',
              'elephant','fox','giant_panda','hamster','horse',
              'lion','lizard','monkey','motorcycle','rabbit',
              'red_panda','sheep','snake','squirrel','tiger',
              'train','turtle','watercraft','whale','zebra']



type_video_dict = {}
for type in type_names:
    # print(type)
    type_video_dict[type] = {}
    type_video_dict[type]['path'] = []
    type_video_dict[type]['track_id'] = []
    type_video_dict[type]['num_frames'] = []
    path_file = os.path.join(file_dir, type, 'path.txt')
    track_id_file = os.path.join(file_dir,type,'track_id.txt')
    num_frames_file = os.path.join(file_dir,type,'num_frames.txt')
    with open(path_file,'r') as f:
        path = f.readlines()
    type_video_dict[type]['path'] = path
    type_video_dict[type]['track_ids'] = np.loadtxt(track_id_file,np.uint8)
    type_video_dict[type]['num_frames'] = np.loadtxt(num_frames_file,np.uint32)
    # assert(len(type_video_dict[type]['track_id']) == len(type_video_dict[type]['num_frames']))
    type_video_dict[type]['num_track_ids'] = len(type_video_dict[type]['track_ids'])
    # print(type,type_video_dict[type]['num_track_ids'])

image_root_dir = os.path.join(VID_ROOT_DIR,'ILSVRC2015/Data/VID/train')
anno_root_dir = os.path.join(VID_ROOT_DIR,'ILSVRC2015/Annotations/VID/train')
def get_roi(type,object_idx,frame_idx,track_id):
    '''instruction'''
    frame_path = os.path.join(image_root_dir, type_video_dict[type]['path'][object_idx][:-1],
                              '%06d.JPEG' % frame_idx)
    frame = cv2.imread(frame_path)
    anno_path = os.path.join(anno_root_dir, type_video_dict[type]['path'][object_idx][:-1],
                             '%06d.xml' % frame_idx)
    tree = ET.parse(anno_path)
    root = tree.getroot()
    objects = root.findall('object')
    if objects != None:
        EXIST = False
        for o in objects:
            tmp = o.find('trackid').text
            if tmp == str(track_id):
                EXIST = True
                # object = objects[track_id] # wrong
                bndbox = o.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                # print(ymin,ymax,xmin,xmax)
                h, w, _ = frame.shape
                xmin = np.clip(xmin, 0, w)
                xmax = np.clip(xmax, 0, w)
                ymin = np.clip(ymin, 0, h)
                ymax = np.clip(ymax, 0, h)
                if (ymax - ymin) >= 5 and (xmax - xmin) >= 5:
                    roi = frame[ymin:ymax, xmin:xmax]
                    standard_roi = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_CUBIC)
                    return standard_roi
        return np.ones((1,))
    else:
        return np.ones((1,))
def get_one_sample(type):
    '''instruction'''
    '''pick up 1 object from a specific type'''
    object_idx = np.random.randint(0, type_video_dict[type]['num_track_ids'], 1)[0]
    track_id = type_video_dict[type]['track_ids'][object_idx]
    track_id_num_frames = type_video_dict[type]['num_frames'][object_idx]
    '''pick up 1 frame from above video'''
    chosen_frame_idx = np.random.randint(0, track_id_num_frames, 1)[0]
    roi = get_roi(type, object_idx, chosen_frame_idx, track_id)
    return roi
def get_two_samples(type):
    '''instruction'''
    '''pick up 1 object from a specific type'''
    object_idx = np.random.randint(0, type_video_dict[type]['num_track_ids'], 1)[0]
    track_id = type_video_dict[type]['track_ids'][object_idx]
    track_id_num_frames = type_video_dict[type]['num_frames'][object_idx]
    '''pick up 2 frame from above video'''
    chosen_frame_idx1,chosen_frame_idx2 = np.random.randint(0, track_id_num_frames, 2)
    roi1 = get_roi(type, object_idx, chosen_frame_idx1, track_id)
    roi2 = get_roi(type, object_idx, chosen_frame_idx2, track_id)
    return (roi1,roi2)
def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

'''save intfrecord'''

for epoch in range(EPOCH):
    tfrecord_name = TFRECORD_PATH + 'VID_epoch%d.tfrecords' % (epoch + 1)
    writer = tf.python_io.TFRecordWriter(tfrecord_name)
    count = 0
    for i in range(PAIRS_PER_EPOCH):
        index = epoch*PAIRS_PER_EPOCH+i
        type1 = type_names[final_result[0,index]]
        type2 = type_names[final_result[1,index]]
        # print(type1,type2)
        A,P = get_two_samples(type1)
        N = get_one_sample(type2)
        if A.ndim == 3 and P.ndim == 3 and N.ndim == 3:
            '''write to tfrecord'''
            A_bytes = A.tobytes()
            P_bytes = P.tobytes()
            N_bytes = N.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'A': bytes_feature(A_bytes),
                'P': bytes_feature(P_bytes),
                'N': bytes_feature(N_bytes)}))

            writer.write(example.SerializeToString())
            count +=1 
            print('%d samples have been saved'%count)
