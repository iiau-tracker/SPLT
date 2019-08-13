#coding=utf-8
from xml.etree import ElementTree as ET
import os

'''Split VID dataset into 30 folders according to class label of the target.'''
# the following paths should be changed to your real path
VID_root = '/data1/Dataset/ILSVRC2015'
save_dir = './train_Skim/type_video'

if __name__ == '__main__':
    VID_anno_root = os.path.join(VID_root, 'Annotations/VID/train')
    folders = ['a','b','c','d','e']
    types = ['n02691156','n02419796','n02131653','n02834778','n01503061',
             'n02924116','n02958343','n02402425','n02084071','n02121808',
             'n02503517','n02118333','n02510455','n02342885','n02374451',
             'n02129165','n01674464','n02484322','n03790512','n02324045',
             'n02509815','n02411705','n01726692','n02355227','n02129604',
             'n04468005','n01662784','n04530566','n02062744','n02391049']
    type_names = ['airplane','antelope','bear','bicycle','bird',
                  'bus','car','cattle','dog','domestic_cat',
                  'elephant','fox','giant_panda','hamster','horse',
                  'lion','lizard','monkey','motorcycle','rabbit',
                  'red_panda','sheep','snake','squirrel','tiger',
                  'train','turtle','watercraft','whale','zebra']
    type_video_dict = {}


    for i in range(30):
        type_video_dict[str(i)] = {}
        type_video_dict[str(i)]['path'] = []
        type_video_dict[str(i)]['track_id'] = []
        type_video_dict[str(i)]['num_frames'] = []

    for folder in folders:
        folder_dir = os.path.join(VID_anno_root,folder)
        videos = sorted(os.listdir(folder_dir))
        for video in videos:
            video_dir = os.path.join(folder_dir,video)
            num_frames = len(os.listdir(video_dir))
            init_xml_file = os.path.join(video_dir,'000000.xml')
            tree = ET.parse(init_xml_file)
            root = tree.getroot()
            '''获取所有object的type与track_id'''
            objects = root.findall('object')
            if objects != None:
                for track_id in range(len(objects)):
                    object = objects[track_id]
                    type = object.find('name').text
                    type_id = types.index(type)
                    type_video_dict[str(type_id)]['path'].append(os.path.join(folder,video))
                    type_video_dict[str(type_id)]['track_id'].append(track_id)
                    type_video_dict[str(type_id)]['num_frames'].append(num_frames)
                    print(folder,video)
    for id in range(30):
        cur_type = type_names[id]
        os.mkdir(save_dir+'/%s' % cur_type)
        path_file = os.path.join(save_dir, '%s/path.txt' % cur_type)
        track_id_file = os.path.join(save_dir, '%s/track_id.txt' % cur_type)
        num_frames_file = os.path.join(save_dir, '%s/num_frames.txt' % cur_type)
        path_list = type_video_dict[str(id)]['path']
        track_id_list = type_video_dict[str(id)]['track_id']
        num_frames_list = type_video_dict[str(id)]['num_frames']
        '''save path file'''
        f1 = open(path_file,'w+')
        for l in path_list:
            f1.write(l)
            f1.write('\n')
        f1.close()
        '''save track_id file'''
        f2 = open(track_id_file,'w+')
        for l in track_id_list:
            f2.write(str(l))
            f2.write('\n')
        f2.close()
        '''save num_frames file'''
        f3 = open(num_frames_file,'w+')
        for l in num_frames_list:
            f3.write(str(l))
            f3.write('\n')
        f3.close()







