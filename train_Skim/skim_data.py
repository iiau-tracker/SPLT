#coding=utf-8
import numpy as np
import os
import cv2
from xml.etree import ElementTree as ET
from PIL import Image
import h5py

'''generate triplet pairs and save in tfrecord for trainning :)'''
file_dir = './train_Skim/type_video'
VID_ROOT_DIR = '/data1/Dataset/ILSVRC2015'

type_names = ['airplane','antelope','bear','bicycle','bird',
              'bus','car','cattle','dog','domestic_cat',
              'elephant','fox','giant_panda','hamster','horse',
              'lion','lizard','monkey','motorcycle','rabbit',
              'red_panda','sheep','snake','squirrel','tiger',
              'train','turtle','watercraft','whale','zebra']

N = 1000
aug_N = int(N * 1.04)
result = np.random.randint(0,30,(2,aug_N)).astype(np.uint8)
mask = (result[0] != result[1])
final_result = result[:,mask][:,:N]


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

image_root_dir = os.path.join(VID_ROOT_DIR, 'Data/VID/train')
anno_root_dir = os.path.join(VID_ROOT_DIR, 'Annotations/VID/train')


def get_T(type,object_idx,frame_idx,track_id):
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
                H, W, _ = frame.shape
                xmin = np.clip(xmin, 0, W)
                xmax = np.clip(xmax, 0, W)
                ymin = np.clip(ymin, 0, H)
                ymax = np.clip(ymax, 0, H)

                w = xmax - xmin
                h = ymax - ymin

                if h >= 30 and w >= 30 and h*1.0/H < 0.7 and w*1.0/W < 0.7:
                    cw = xmin + w / 2
                    ch = ymin + h / 2

                    half_w = int(w / 2 * 1.3)
                    half_h = int(h / 2 * 1.3)

                    crop = np.array([cw - half_w, ch - half_h, cw + half_w, ch + half_h], dtype=int)

                    img = Image.fromarray(frame)
                    img = img.crop(crop)
                    img = img.resize([140, 140])
                    img = np.array(img).astype('float32')

                    return img
        return None
    else:
        return None


def get_S(type, object_idx, frame_idx, track_id):
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
                H, W, _ = frame.shape
                xmin = np.clip(xmin, 0, W-1)
                xmax = np.clip(xmax, 0, W-1)
                ymin = np.clip(ymin, 0, H-1)
                ymax = np.clip(ymax, 0, H-1)

                w = xmax - xmin
                h = ymax - ymin

                if h >= 30 and w >= 30 and h * 1.0 / H < 0.7 and w * 1.0 / W < 0.7:
                    if np.random.rand() > 0.5:
                        frame[ymin:ymax, xmin:xmax, :] = np.array([np.mean(frame[:,:,0]),np.mean(frame[:,:,1]), np.mean(frame[:,:,2])], dtype='uint8')
                        label = 0
                    else:
                        label = 1


                    half = int((w + h) / 2 * 2.4)

                    cw = xmin + w / 2 + np.random.randint(-half*0.5, half*0.5)
                    ch = ymin + h / 2 + np.random.randint(-half*0.5, half*0.5)

                    crop = np.array([cw - half, ch - half, cw + half, ch + half], dtype=int)

                    img = Image.fromarray(frame)
                    img = img.crop(crop)
                    img = img.resize([256, 256])
                    img = np.array(img).astype('float32')

                    return img, label
        return None, None
    else:
        return None, None



def get_two_samples(type):
    '''instruction'''
    '''pick up 1 object from a specific type'''
    object_idx = np.random.randint(0, type_video_dict[type]['num_track_ids'], 1)[0]
    track_id = type_video_dict[type]['track_ids'][object_idx]
    track_id_num_frames = type_video_dict[type]['num_frames'][object_idx]
    '''pick up 2 frame from above video'''
    chosen_frame_idx1,chosen_frame_idx2 = np.random.randint(0, track_id_num_frames, 2)
    roi1 = get_T(type, object_idx, chosen_frame_idx1, track_id)
    roi2, label = get_S(type, object_idx, chosen_frame_idx2, track_id)
    return (roi1, roi2, label)


num = 0
summ = 4000
save = './Siam/Skim_data.h5'

fdata = h5py.File(save, 'w')
fdata.create_dataset('template',(summ,140, 140, 3), dtype='float32')
fdata.create_dataset('search',(summ, 256, 256, 3), dtype='float32')
fdata.create_dataset('label',(summ, 1), dtype='float32')

while num < summ:
    type = type_names[final_result[0, num % 500]]
    template, search, label = get_two_samples(type)
    if search is not  None and template is not None:
        fdata['search'][num] = search
        fdata['template'][num] = template
        fdata['label'][num] = label

        num += 1
        print num, label, type
fdata.close()
