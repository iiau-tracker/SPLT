# coding=utf-8
import tensorflow as tf
import os
import _init_paths
from config import cfg
import cv2
import numpy as np
import random


'''VID'''
DATA_PATH = '/SSD/VID_tfrecord/'
TFRecord_list = os.listdir(DATA_PATH)
full_TFRecord_list = [os.path.realpath(DATA_PATH + file) for file in TFRecord_list]


random.shuffle(full_TFRecord_list)
TIMES = 10
# cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.BATCH_SIZE = 1
mean = tf.constant([122.6789, 116.6688, 104.0069], dtype=tf.float32, name='mean', shape=(1, 1, 3))
mean_arr = np.reshape(np.array([122.6789, 116.6688, 104.0069]),(1,1,3))
###################################################
def preprocess(A,P,N):

    A = tf.cast(A, tf.float32)
    P = tf.cast(P, tf.float32)
    N = tf.cast(N, tf.float32)

    # A = tf.image.random_flip_left_right(A)
    # P = tf.image.random_flip_left_right(P)
    # N = tf.image.random_flip_left_right(N)

    A = tf.subtract(A,mean,name='A')
    P = tf.subtract(P,mean,name='P')
    N = tf.subtract(N,mean,name='N')
    return A,P,N


def get_inputs():
    dataset = tf.data.TFRecordDataset(full_TFRecord_list)
    def parser(record):
        features = tf.parse_single_example(
            record,
            features={
                'A': tf.FixedLenFeature([], tf.string),
                'P': tf.FixedLenFeature([], tf.string),
                'N': tf.FixedLenFeature([], tf.string)})
        A_bytes = features['A']
        P_bytes = features['P']
        N_bytes = features['N']
        '''must be decoded as uint8'''
        A = tf.decode_raw(A_bytes,tf.uint8)
        P = tf.decode_raw(P_bytes, tf.uint8)
        N = tf.decode_raw(N_bytes, tf.uint8)
        # A.set_shape([128, 128, 3])
        # P.set_shape([128, 128, 3])
        # N.set_shape([128, 128, 3])
        A = tf.reshape(A, (128, 128, 3))
        P = tf.reshape(P, (128, 128, 3))
        N = tf.reshape(N, (128, 128, 3))
        # img2_input.set_shape([1, 512, 512, 3])
        A,P,N = preprocess(A,P,N)
        return A,P,N

    '''dataset method will automatically add a dim'''
    dataset = dataset.map(parser)
    # dataset = dataset.shuffle(buffer_size = 100)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(TIMES)
    dataset = dataset.batch(cfg.TRAIN.BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    A,P,N = iterator.get_next()
    return A,P,N
'''computational graph'''
# #
# A_tensor,P_tensor,N_tensor = get_inputs()
#
# with tf.Session() as sess:
#     for i in range(60000*50):
#         A_array,P_array,N_array = sess.run([A_tensor,P_tensor,N_tensor])
#         A = A_array.squeeze()
#         P = P_array.squeeze()
#         N = N_array.squeeze()
#         A += mean_arr
#         P += mean_arr
#         N += mean_arr
#         A = A.astype(np.uint8)
#         P = P.astype(np.uint8)
#         N = N.astype(np.uint8)
#         # print(A.shape)
#         # print(P.shape)
#         # print(N.shape)
#         print(i)
#         whole_image = np.concatenate((A,P,N),axis=1)
#         cv2.imshow('main_window',whole_image)
#         cv2.waitKey(500)


