import os
import sys

PROJECT_PATH = '/home/masterbin-iiau/SPLT/'
sys.path.append(PROJECT_PATH + 'lib')
sys.path.append(PROJECT_PATH + 'lib/slim')

import cv2
import random
import numpy as np
from time import time

import vot
# import torch
import tensorflow as tf

from utils.tracking_utils import build_init_graph, build_box_predictor, restore_model, get_configs_from_pipeline_file
from utils.tracking_utils import crop_search_region, gen_search_patch_Hao, crop_template_Hao
from utils.tracking_utils import show_res, compile_results

from core.model_builder import build_man_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)


V_NET = 'resnet50'#mobilenet,vgg16,resnet50
V_OPTION = 'VID_N'
S_NET = 'M' 

V_T = 0.7  # 2-2theta = 0.7, theta = 0.65
G_V_T = 0.8

O_T_L = 0.3 
O_T_H = 0.7
O_T_C = 0.6


if V_NET == 'resnet101':
    iteration = 131249
if V_NET == 'resnet50':
    iteration = 65624
if V_NET == 'mobilenet':
    iteration = 8202
v_name = 'V_%s_%s-%d'%(V_NET,V_OPTION,iteration)
pretrained_model = os.path.join(PROJECT_PATH,'Verifier/ckpt/',v_name)

if V_NET == 'resnet101':
    from Verifier.resnet101_bin import _image_to_feat
if V_NET == 'resnet50':
    from Verifier.resnet50_bin import _image_to_feat
elif V_NET == 'mobilenet':
    from Verifier.mobilenet import _image_to_feat
elif V_NET == 'vgg16':
    from Verifier.vgg16 import _image_to_feat



class MobileTracker(object):
    def __init__(self, vot=False, dis=False):
        self.vot = vot
        self.dis = dis


        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.V_image_op = tf.placeholder(tf.float32, (None, 128, 128, 3), name='V_input_image')
        self.V_feat_op = _image_to_feat(self.V_image_op, is_training=False, reuse=False)

        variables = tf.global_variables()  # list
        # Initialize all variables first
        self.sess.run(tf.variables_initializer(variables, name='init')) 

        restorer = tf.train.Saver(variables)
        restorer.restore(self.sess, pretrained_model)


        config_file = PROJECT_PATH + 'RPN/ssd_mobilenet_tracking.config'
        checkpoint_dir = PROJECT_PATH + 'RPN/dump'

        model_config, train_config, input_config, eval_config \
            = get_configs_from_pipeline_file(config_file)

        model_scope = 'model'
        model = build_man_model(model_config=model_config, is_training=False)

        self.initConstantOp = tf.placeholder(tf.float32, [1, 1, 1, 512])

        self.initFeatOp, self.initInputOp \
            = build_init_graph(model, model_scope, reuse=None)

        self.pre_box_tensor, self.scores_tensor, self.input_cur_image \
            = build_box_predictor(model, model_scope, self.initConstantOp, reuse=None)


        # self.sess.run(tf.global_variables_initializer())
        variables_to_restore = tf.global_variables()
        restore_model(self.sess, model_scope, checkpoint_dir, variables_to_restore, V_NET)


        with tf.keras.utils.CustomObjectScope({
            'relu6': tf.keras.layers.LeakyReLU(0),
            'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D}):
            self.branch_z = tf.keras.models.load_model(
                PROJECT_PATH + 'Skim/branch_z.h5',
                custom_objects={"tf": tf})
            self.branch_search = tf.keras.models.load_model(
                PROJECT_PATH + 'Skim/branch_search.h5',
                custom_objects={"tf": tf})

    def init_first(self, image, region):


        # region.x region.y region.width region.height
        if self.vot:
            gt_for_siamfc = np.array([
                region.x,
                region.y,
                region.width,
                region.height
            ]).astype(int)

            init_gt = [
                region.y,
                region.x,
                region.y + region.height,
                region.x + region.width
            ]  # ymin xmin ymax xmax
        else:
            gt_for_siamfc = np.array([
                region[0],
                region[1],
                region[2],
                region[3]
            ]).astype(int)

            init_gt = [
                region[1],
                region[0],
                region[1] + region[3],
                region[0] + region[2]
            ]  # ymin xmin ymax xmax

        self.expand_channel = False

        if image.ndim < 3:
            image = np.expand_dims(image, axis=2)
            image = np.repeat(image, repeats=3, axis=2)
            init_img = image
            self.expand_channel = True
        else:
            init_img = image


        gt_boxes = np.zeros((1, 4))
        gt_boxes[0, 0] = init_gt[0] / float(init_img.shape[0])
        gt_boxes[0, 1] = init_gt[1] / float(init_img.shape[1])
        gt_boxes[0, 2] = init_gt[2] / float(init_img.shape[0])
        gt_boxes[0, 3] = init_gt[3] / float(init_img.shape[1])


        pad_x = 36.0 / 264.0 * (gt_boxes[0, 3] - gt_boxes[0, 1]) * init_img.shape[1]
        pad_y = 36.0 / 264.0 * (gt_boxes[0, 2] - gt_boxes[0, 0]) * init_img.shape[0]
        cx = (gt_boxes[0, 3] + gt_boxes[0, 1]) / 2.0 * init_img.shape[1]
        cy = (gt_boxes[0, 2] + gt_boxes[0, 0]) / 2.0 * init_img.shape[0]
        startx = gt_boxes[0, 1] * init_img.shape[1] - pad_x
        starty = gt_boxes[0, 0] * init_img.shape[0] - pad_y
        endx = gt_boxes[0, 3] * init_img.shape[1] + pad_x
        endy = gt_boxes[0, 2] * init_img.shape[0] + pad_y
        left_pad = max(0, int(-startx))
        top_pad = max(0, int(-starty))
        right_pad = max(0, int(endx - init_img.shape[1] + 1))
        bottom_pad = max(0, int(endy - init_img.shape[0] + 1))

        startx = int(startx + left_pad)
        starty = int(starty + top_pad)
        endx = int(endx + left_pad)
        endy = int(endy + top_pad)

        img1_xiaobai = init_img.copy()
        if top_pad or left_pad or bottom_pad or right_pad:
            r = np.pad(
                img1_xiaobai[:, :, 0],
                ((top_pad, bottom_pad), (left_pad, right_pad)),
                mode='constant', constant_values=128)

            g = np.pad(
                img1_xiaobai[:, :, 1],
                ((top_pad, bottom_pad), (left_pad, right_pad)),
                mode='constant', constant_values=128)

            b = np.pad(
                img1_xiaobai[:, :, 2],
                ((top_pad, bottom_pad), (left_pad, right_pad)),
                mode='constant', constant_values=128)

            r = np.expand_dims(r, 2)
            g = np.expand_dims(g, 2)
            b = np.expand_dims(b, 2)

            img1_xiaobai = np.concatenate((r, g, b), axis=2)

        # gt_boxes resize

        init_img_crop = img1_xiaobai[starty:endy, startx:endx]
        init_img_crop = cv2.resize(init_img_crop, (128, 128))

        self.last_gt = init_gt
        self.init_feature_maps = self.sess.run(
            self.initFeatOp,
            feed_dict={self.initInputOp: init_img_crop})


        self.mean = np.reshape(np.array([122.6789, 116.6688, 104.0069]), (1, 1, 3))

        ori_ymin = int(init_gt[0])
        ori_xmin = int(init_gt[1])
        ori_ymax = int(init_gt[2] + 1)
        ori_xmax = int(init_gt[3] + 1)

        unscaled_win = image[ori_ymin:ori_ymax, ori_xmin:ori_xmax]
        template_image = cv2.resize(unscaled_win, (128, 128)).astype(np.float64)
        template_image -= self.mean
        template_image_ = template_image[np.newaxis, :]
        self.template_feat = self.sess.run(
            self.V_feat_op,
            feed_dict={self.V_image_op: template_image_})


        z_im = crop_template_Hao(init_img, gt_for_siamfc)

        z_im = z_im[None, :, :, :].astype(np.float32)
        self.z_feat = self.branch_z.predict(z_im)
        # warm
        self.branch_search.predict([self.z_feat.repeat(1, axis=0), np.random.rand(1,256,256,3)])


        self.V_thres = V_T 
        self.global_V_thres = G_V_T  

        self.Object_thres_low = O_T_L  
        self.Object_thres_high = O_T_H  
        self.Object_thres_center = O_T_C  

        self.EXTREM = 0.02 
        self.LargeDist = 100 
        self.k = 20 
        self.target_w = init_gt[3] - init_gt[1]
        self.target_h = init_gt[2] - init_gt[0]
        self.first_w = init_gt[3] - init_gt[1]
        self.first_h = init_gt[2] - init_gt[0]
        self.i = 0
        self.startx = 0
        self.starty = 0

        self.SEARCH_K = 4 # k = search_k -1

    def center_search(
            self,
            image_,
            base_h, base_w,
            ori_scores, ori_best_idx, ori_detection_box, ori_dist_min):

        # search_gt = (np.array(self.last_gt)).copy()
        search_gt = np.zeros((4,))
        search_gt[0] = image_.shape[0] / 2.0 - base_h / 2.0
        search_gt[2] = image_.shape[0] / 2.0 + base_h / 2.0
        search_gt[1] = image_.shape[1] / 2.0 - base_w / 2.0
        search_gt[3] = image_.shape[1] / 2.0 + base_w / 2.0

        cur_img_array, win_loc1, scale1 = \
            crop_search_region(image_, search_gt, 300, mean_rgb=128)

        detection_box_ori1, scores1 = self.sess.run(
            [self.pre_box_tensor, self.scores_tensor],
            feed_dict={
                self.input_cur_image: cur_img_array,
                self.initConstantOp: self.init_feature_maps})

        if scores1[0, 0] > self.Object_thres_center:
            detection_box_ori1[:, 0] = detection_box_ori1[:, 0] * scale1[0] + win_loc1[0]
            detection_box_ori1[:, 1] = detection_box_ori1[:, 1] * scale1[1] + win_loc1[1]
            detection_box_ori1[:, 2] = detection_box_ori1[:, 2] * scale1[0] + win_loc1[0]
            detection_box_ori1[:, 3] = detection_box_ori1[:, 3] * scale1[1] + win_loc1[1]
            detection_box_ori = detection_box_ori1.copy()
            # max_idx = 0
            search_box1 = detection_box_ori[0]

            search_box1[0] = np.clip(search_box1[0], 0, image_.shape[0] - 1)
            search_box1[2] = np.clip(search_box1[2], 0, image_.shape[0] - 1)
            search_box1[1] = np.clip(search_box1[1], 0, image_.shape[1] - 1)
            search_box1[3] = np.clip(search_box1[3], 0, image_.shape[1] - 1)

            if (int(search_box1[0]) == int(search_box1[2])
                    or int(search_box1[1]) == int(search_box1[3])):
                dist_min = self.LargeDist
            else:
                unscaled_win = image_[
                                int(search_box1[0]):int(search_box1[2]),
                                int(search_box1[1]):int(search_box1[3])]

                win = cv2.resize(unscaled_win, (128, 128)).astype(np.float64)
                win -= self.mean
                win_input = win[np.newaxis, :]

                candidate_feat = self.sess.run(
                    self.V_feat_op,
                    feed_dict={self.V_image_op: win_input})

                dist_min = np.sum(np.square(self.template_feat - candidate_feat))

            if dist_min < self.V_thres:

                best_idx = 0
                scores = scores1.copy()
                detection_box = detection_box_ori[best_idx]
            else:
                search_box1 = detection_box_ori[:self.k]
                search_box = np.zeros_like(search_box1)  # x1 y1 x2 y2
                search_box[:, 0] = search_box1[:, 1]
                search_box[:, 1] = search_box1[:, 0]
                search_box[:, 2] = search_box1[:, 3]
                search_box[:, 3] = search_box1[:, 2]
                search_box[:, 2] = search_box[:, 2] - search_box[:, 0]  # x y w h
                search_box[:, 3] = search_box[:, 3] - search_box[:, 1]


                search_box[:, 2] = np.maximum(search_box[:, 2], 3)
                search_box[:, 3] = np.maximum(search_box[:, 3], 3)

                search_box[:, 0] = np.maximum(search_box[:, 0], 0)
                search_box[:, 1] = np.maximum(search_box[:, 1], 0)

                search_box[:, 0] = np.minimum(
                    search_box[:, 0],
                    image_.shape[1] - search_box[:, 2] - 1)

                search_box[:, 1] = np.minimum(
                    search_box[:, 1],
                    image_.shape[0] - search_box[:, 3] - 1)

                ID = np.arange(self.k)
                O_mask = (scores1[0, :self.k] > self.Object_thres_low)
                ID_obj = ID[O_mask]
                num_object = int(np.sum(O_mask))

                win_input = np.zeros((num_object, 128, 128, 3))

                starty = search_box[O_mask, 1]
                startx = search_box[O_mask, 0]
                endy = search_box[O_mask, 3] + search_box[O_mask, 1]
                endx = search_box[O_mask, 2] + search_box[O_mask, 0]

                for i in range(num_object):
                    unscaled_win = image_[
                                   int(starty[i]):int(endy[i]),
                                   int(startx[i]):int(endx[i])]
                    win_input[i] = cv2.resize(unscaled_win, (128, 128)).astype(np.float64)

                win_input = win_input - self.mean.reshape((1, 1, 1, 3))

                candidate_feats = self.sess.run(
                    self.V_feat_op,
                    feed_dict={self.V_image_op: win_input})

                dists = np.sum(np.square(self.template_feat - candidate_feats), axis=-1)
                min_idx1 = np.argmin(dists)

                if (dists[min_idx1] < self.V_thres
                        and scores1[0, ID_obj[min_idx1]] > self.Object_thres_high):

                    dist_min = dists[min_idx1]
                    best_idx = ID_obj[min_idx1]
                    scores = scores1.copy()
                    detection_box = detection_box_ori[best_idx]
                else:

                    return ori_scores, ori_best_idx, ori_detection_box, ori_dist_min
                    # detection_box = detection_box_ori[max_idx]
            return scores, best_idx, detection_box, dist_min
        else:
            return ori_scores, ori_best_idx, ori_detection_box, ori_dist_min



    def track(self, image):
        self.i += 1


        cur_ori_img = image

        cur_img_array, win_loc, scale \
            = crop_search_region(cur_ori_img, self.last_gt, 300, mean_rgb=128)


        detection_box_ori, scores = self.sess.run(
            [self.pre_box_tensor, self.scores_tensor],
            feed_dict={
                self.input_cur_image: cur_img_array,
                self.initConstantOp: self.init_feature_maps})


        detection_box_ori[:, 0] = detection_box_ori[:, 0] * scale[0] + win_loc[0]
        detection_box_ori[:, 1] = detection_box_ori[:, 1] * scale[1] + win_loc[1]
        detection_box_ori[:, 2] = detection_box_ori[:, 2] * scale[0] + win_loc[0]
        detection_box_ori[:, 3] = detection_box_ori[:, 3] * scale[1] + win_loc[1]

        A_candis = ((detection_box_ori[:self.k, 3] - detection_box_ori[:self.k, 1])
                    * (detection_box_ori[:self.k, 2] - detection_box_ori[:self.k, 0]))

        A_lastgt = ((self.last_gt[3] - self.last_gt[1])
                    * (self.last_gt[2] - self.last_gt[0]))
        x1 = np.maximum(detection_box_ori[:self.k, 1], self.last_gt[1])
        y1 = np.maximum(detection_box_ori[:self.k, 0], self.last_gt[0])
        x2 = np.minimum(detection_box_ori[:self.k, 3], self.last_gt[3])
        y2 = np.minimum(detection_box_ori[:self.k, 2], self.last_gt[2])
        inter = np.maximum((x2 - x1), 0) * np.maximum((y2 - y1), 0)
        IOU = inter / (A_candis + A_lastgt - inter)
        ID = np.arange(self.k)

        threshold = 0.4
        I_mask = IOU > threshold
        ID_iou = ID[I_mask]

        if np.sum(I_mask) > 0:

            best_idx = ID_iou[np.argmax(scores[0, :self.k][I_mask])] 
        else:
            best_idx = 0  

        search_box1 = detection_box_ori[best_idx]
        search_box1[0] = np.clip(search_box1[0], 0, cur_ori_img.shape[0] - 1)
        search_box1[2] = np.clip(search_box1[2], 0, cur_ori_img.shape[0] - 1)
        search_box1[1] = np.clip(search_box1[1], 0, cur_ori_img.shape[1] - 1)
        search_box1[3] = np.clip(search_box1[3], 0, cur_ori_img.shape[1] - 1)

        if (int(search_box1[0]) == int(search_box1[2])
                or int(search_box1[1]) == int(search_box1[3])):
            dist_min = self.LargeDist  
        else:
            unscaled_win = image[
                           int(search_box1[0]):int(search_box1[2]),
                           int(search_box1[1]):int(search_box1[3])]
            win = cv2.resize(unscaled_win, (128, 128)).astype(np.float64)
            win -= self.mean
            win_input = win[np.newaxis, :]
            candidate_feat = self.sess.run(
                self.V_feat_op,
                feed_dict={self.V_image_op: win_input})

            dist_min = np.sum(np.square(self.template_feat - candidate_feat)) 

        # if score_max < self.classi_threshold:

        if dist_min > self.V_thres:
            search_box1 = detection_box_ori[:self.k]
            search_box = np.zeros_like(search_box1)  # x1 y1 x2 y2
            search_box[:, 0] = search_box1[:, 1]
            search_box[:, 1] = search_box1[:, 0]
            search_box[:, 2] = search_box1[:, 3]
            search_box[:, 3] = search_box1[:, 2]
            search_box[:, 2] = search_box[:, 2] - search_box[:, 0]  # x y w h
            search_box[:, 3] = search_box[:, 3] - search_box[:, 1]


            search_box[:, 2] = np.maximum(search_box[:, 2], 3)
            search_box[:, 3] = np.maximum(search_box[:, 3], 3)

            search_box[:, 0] = np.maximum(search_box[:, 0], 0)
            search_box[:, 1] = np.maximum(search_box[:, 1], 0)

            search_box[:, 0] = np.minimum(
                search_box[:, 0],
                cur_ori_img.shape[1] - search_box[:, 2] - 1)

            search_box[:, 1] = np.minimum(
                search_box[:, 1],
                cur_ori_img.shape[0] - search_box[:, 3] - 1)

            if scores[0, 0] > self.Object_thres_low:
                O_mask = (scores[0, :self.k] > self.Object_thres_low)
                ID_obj = ID[O_mask]
                num_object = int(np.sum(O_mask))

                win_input = np.zeros((num_object, 128, 128, 3))

                starty = search_box[O_mask, 1]
                startx = search_box[O_mask, 0]
                endy = search_box[O_mask, 3] + search_box[O_mask, 1]
                endx = search_box[O_mask, 2] + search_box[O_mask, 0]

                for i in range(num_object):
                    unscaled_win = image[
                                   int(starty[i]):int(endy[i]),
                                   int(startx[i]):int(endx[i])]
                    win_input[i] = cv2.resize(unscaled_win, (128, 128)).astype(np.float64)

                win_input = win_input - self.mean.reshape((1, 1, 1, 3))
                candidate_feats = self.sess.run(
                    self.V_feat_op,
                    feed_dict={self.V_image_op: win_input})
                dists = np.sum(np.square(self.template_feat - candidate_feats), axis=-1)


                dists1 = dists.copy()
                for i in range(num_object):
                    if ID_obj[i] not in ID_iou:
                        dists1[i] = self.LargeDist  # IOU < threshold

                if np.min(dists1) < self.V_thres:

                    best_idx = ID_obj[np.argmin(dists1)]
                    dist_min = np.min(dists1)
                elif np.min(dists) < self.V_thres:

                    best_idx = ID_obj[np.argmin(dists)]
                    dist_min = np.min(dists)
                else:

                    dist_min = self.LargeDist

        detection_box = detection_box_ori[best_idx]


        if scores[0, best_idx] < self.Object_thres_low:

            scores, best_idx, detection_box, dist_min \
                = self.center_search(
                cur_ori_img,
                (self.last_gt[2] - self.last_gt[0]),
                (self.last_gt[3] - self.last_gt[1]),
                scores, best_idx, detection_box, dist_min)

            if dist_min > self.V_thres:

                scores, best_idx, detection_box, dist_min \
                    = self.center_search(
                    cur_ori_img,
                    self.first_h,
                    self.first_w,
                    scores, best_idx, detection_box, dist_min)

            if dist_min > self.V_thres:

                scores, best_idx, detection_box, dist_min \
                    = self.center_search(
                    cur_ori_img,
                    self.first_h / 2.0,
                    self.first_w / 2.0,
                    scores, best_idx, detection_box, dist_min)

            if dist_min > self.V_thres:

                scores, best_idx, detection_box, dist_min \
                    = self.center_search(
                    cur_ori_img,
                    self.first_h * 2.0,
                    self.first_w * 2.0,
                    scores, best_idx, detection_box, dist_min)


        # print scores[0,max_idx]
        if scores[0, best_idx] < self.Object_thres_low:
            """-------------------------------------------------------------------------"""
            softmax_test_, pos_i = gen_search_patch_Hao(cur_ori_img, self.first_w, self.first_h)
            softmax_test = softmax_test_.astype(np.float32)
            print 'global'
            batch_sz = 64

            if softmax_test.shape[0] <= batch_sz:
                kk = softmax_test
                cls_out = self.branch_search.predict([self.z_feat.repeat(kk.shape[0], axis=0), kk]).reshape(-1)


            elif softmax_test.shape[0] > batch_sz:
                cls_out_list = []

                for_i = softmax_test.shape[0] / batch_sz
                for jj in range(for_i):
                    kk = softmax_test[batch_sz * jj:batch_sz * (jj + 1)]
                    cls_out_list.append(
                        self.branch_search.predict([self.z_feat.repeat(kk.shape[0], axis=0), kk]).reshape(-1) )

                if softmax_test.shape[0] % batch_sz == 0:
                    pass
                else:
                    kk = softmax_test[batch_sz * (jj + 1):]
                    cls_out_list.append(
                        self.branch_search.predict([self.z_feat.repeat(kk.shape[0], axis=0), kk]).reshape(-1) )

                cls_out = np.concatenate(cls_out_list)

            search_rank = np.argsort(-cls_out)
            pos_i = pos_i[search_rank]
            cls_out = cls_out[search_rank]
            """-------------------------------------------------------------------------"""

            self.SEARCH_K = np.minimum(pos_i.shape[0], self.SEARCH_K)
            if self.SEARCH_K > 1:
                search_num = self.SEARCH_K - 1
            else:
                search_num = 1

            detection_box1_all =  np.zeros([search_num,4])
            scores1_all =  np.zeros([1,search_num])

            for s_i in range(search_num):
                search_gt = pos_i[s_i]

                cur_img_array1, win_loc1, scale1 \
                    = crop_search_region(cur_ori_img, search_gt, 300, mean_rgb=128)
                detection_box1, scores1 = self.sess.run(
                    [self.pre_box_tensor, self.scores_tensor],
                    feed_dict={
                        self.input_cur_image: cur_img_array1,
                        self.initConstantOp: self.init_feature_maps})

                detection_box1[0, 0] = detection_box1[0, 0] * scale1[0] + win_loc1[0]
                detection_box1[0, 1] = detection_box1[0, 1] * scale1[1] + win_loc1[1]
                detection_box1[0, 2] = detection_box1[0, 2] * scale1[0] + win_loc1[0]
                detection_box1[0, 3] = detection_box1[0, 3] * scale1[1] + win_loc1[1]

                scores1_all[0,s_i] = scores1[0, 0]
                detection_box1_all[s_i] = detection_box1[0].copy()

            rank_idx = np.argsort(-scores1_all).reshape(-1)
            scores1 = scores1_all[:,rank_idx]
            detection_box1 = detection_box1_all[rank_idx,:]

            if scores1[0, 0] > self.Object_thres_high:
                detection_box_ori = detection_box1.copy()
                # max_idx = 0
                search_box1 = detection_box_ori[0]
                search_box1[0] = np.clip(search_box1[0], 0, cur_ori_img.shape[0] - 1)
                search_box1[2] = np.clip(search_box1[2], 0, cur_ori_img.shape[0] - 1)
                search_box1[1] = np.clip(search_box1[1], 0, cur_ori_img.shape[1] - 1)
                search_box1[3] = np.clip(search_box1[3], 0, cur_ori_img.shape[1] - 1)
                if (int(search_box1[0]) == int(search_box1[2])
                        or int(search_box1[1]) == int(search_box1[3])):
                    # score_max = -1
                    # score_max = 0  # 0 is the minimum score for SINT
                    dist_min = self.LargeDist
                else:
                    search_box1 = [
                        search_box1[1], search_box1[0],
                        search_box1[3] - search_box1[1],
                        search_box1[2] - search_box1[0]]

                    search_box1 = np.reshape(search_box1, (4,))

                    unscaled_win = image[
                                    int(search_box1[1]):int(search_box1[3] + search_box1[1]),
                                    int(search_box1[0]):int(search_box1[2] + search_box1[0])]

                    win = cv2.resize(unscaled_win, (128, 128)).astype(np.float64)
                    win -= self.mean
                    win_input = win[np.newaxis, :]
                    candidate_feat = self.sess.run(
                        self.V_feat_op,
                        feed_dict={self.V_image_op: win_input})

                    dist_min = np.sum(np.square(self.template_feat - candidate_feat))

                if dist_min < self.global_V_thres:

                    scores = scores1.copy()
                    best_idx = 0
                    detection_box = detection_box_ori[best_idx]

                elif dist_min > self.global_V_thres and self.SEARCH_K-search_num > 0:

                    search_gt = pos_i[search_num]

                    cur_img_array1, win_loc1, scale1 \
                        = crop_search_region(cur_ori_img, search_gt, 300, mean_rgb=128)
                    detection_box1, scores1 = self.sess.run(
                        [self.pre_box_tensor, self.scores_tensor],
                        feed_dict={
                            self.input_cur_image: cur_img_array1,
                            self.initConstantOp: self.init_feature_maps})

                    detection_box1[0, 0] = detection_box1[0, 0] * scale1[0] + win_loc1[0]
                    detection_box1[0, 1] = detection_box1[0, 1] * scale1[1] + win_loc1[1]
                    detection_box1[0, 2] = detection_box1[0, 2] * scale1[0] + win_loc1[0]
                    detection_box1[0, 3] = detection_box1[0, 3] * scale1[1] + win_loc1[1]
                    detection_box_ori = detection_box1.copy()
                    # max_idx = 0
                    search_box1 = detection_box_ori[0]
                    search_box1[0] = np.clip(search_box1[0], 0, cur_ori_img.shape[0] - 1)
                    search_box1[2] = np.clip(search_box1[2], 0, cur_ori_img.shape[0] - 1)
                    search_box1[1] = np.clip(search_box1[1], 0, cur_ori_img.shape[1] - 1)
                    search_box1[3] = np.clip(search_box1[3], 0, cur_ori_img.shape[1] - 1)
                    if (int(search_box1[0]) == int(search_box1[2])
                            or int(search_box1[1]) == int(search_box1[3])):
                        dist_min = self.LargeDist
                    else:
                        search_box1 = [
                            search_box1[1], search_box1[0],
                            search_box1[3] - search_box1[1],
                            search_box1[2] - search_box1[0]]

                        search_box1 = np.reshape(search_box1, (4,))

                        unscaled_win = image[
                                        int(search_box1[1]):int(search_box1[3] + search_box1[1]),
                                        int(search_box1[0]):int(search_box1[2] + search_box1[0])]

                        win = cv2.resize(unscaled_win, (128, 128)).astype(np.float64)
                        win -= self.mean
                        win_input = win[np.newaxis, :]
                        candidate_feat = self.sess.run(
                            self.V_feat_op,
                            feed_dict={self.V_image_op: win_input})

                        dist_min = np.sum(np.square(self.template_feat - candidate_feat))

                    if dist_min < self.global_V_thres:

                        scores = scores1.copy()
                        best_idx = 0
                        detection_box = detection_box_ori[best_idx]



        if scores[0, best_idx] < self.Object_thres_low:

            x_c = (detection_box[3] + detection_box[1]) / 2.0
            y_c = (detection_box[0] + detection_box[2]) / 2.0
            w1 = self.last_gt[3] - self.last_gt[1]
            h1 = self.last_gt[2] - self.last_gt[0]
            x1 = x_c - w1 / 2.0
            y1 = y_c - h1 / 2.0
            x2 = x_c + w1 / 2.0
            y2 = y_c + h1 / 2.0
            self.last_gt = np.float32([y1, x1, y2, x2])
        else:

            self.last_gt = detection_box
            self.target_w = detection_box[3] - detection_box[1]
            self.target_h = detection_box[2] - detection_box[0]

        if self.last_gt[0] < 0:
            self.last_gt[0] = 0
            self.last_gt[2] = self.target_h
        if self.last_gt[1] < 0:
            self.last_gt[1] = 0
            self.last_gt[3] = self.target_w
        if self.last_gt[2] > cur_ori_img.shape[0]:
            self.last_gt[2] = cur_ori_img.shape[0] - 1
            self.last_gt[0] = cur_ori_img.shape[0] - 1 - self.target_h
        if self.last_gt[3] > cur_ori_img.shape[1]:
            self.last_gt[3] = cur_ori_img.shape[1] - 1
            self.last_gt[1] = cur_ori_img.shape[1] - 1 - self.target_w

        self.target_w = (self.last_gt[3] - self.last_gt[1])
        self.target_h = (self.last_gt[2] - self.last_gt[0])

        width = self.last_gt[3] - self.last_gt[1]
        height = self.last_gt[2] - self.last_gt[0]

        if self.dis:
            show_res(
                image,
                np.array(self.last_gt, dtype=np.int32),
                '2',
                score=scores[0,best_idx],
                score_max=dist_min)

        if (scores[0, best_idx] > self.Object_thres_high
                and dist_min < self.V_thres):

            confidence_score = 0.99

        elif (scores[0, best_idx] < self.Object_thres_low
              and dist_min > self.V_thres):

            confidence_score = np.nan

        elif dist_min < self.EXTREM:

            confidence_score = 0.99

        else:
            confidence_score = scores[0, best_idx]


        if self.vot:
            return vot.Rectangle(
                float(self.last_gt[1]),
                float(self.last_gt[0]),
                float(width),
                float(height)
            ), confidence_score
        else:
            return np.array([
                float(self.last_gt[1]),
                float(self.last_gt[0]),
                float(width),
                float(height)
            ]), confidence_score



use_vot = True
display = False
save_results = False

# use_vot = False
# display = True
# save_results = False


if use_vot:
    """VOT """
    handle = vot.VOT("rectangle")
    selection = handle.region()
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)

    image = cv2.imread(imagefile)
    # print image.shape
    tracker = MobileTracker(vot=use_vot, dis=display)
    tracker.init_first(image, selection)

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break

        image = cv2.imread(imagefile)
        region, confidence = tracker.track(image)
        handle.report(region, confidence)


