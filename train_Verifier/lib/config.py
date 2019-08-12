#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from config import cfg
cfg = __C

#
# Training options
#
__C.TRAIN = edict()

__C.TRAIN.NUM_GPUS = 2

# __C.TRAIN.IMAGE_ROOT = '/media/masterbin-iiau/417507b2-6b1b-4e11-b488-4e0ff0e89a481/tangjiuqi097/ILSVRC2015_VID/ILSVRC2015/Data/VID/train/'
# Initial learning rate
__C.TRAIN.LEARNING_RATE = 1e-2

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 5e-4

# BATCH_SIZE for training
__C.TRAIN.BATCH_SIZE = 32
# margin
__C.TRAIN.MARGIN = 0.2

# 训练多少个epoches
__C.TRAIN.EPOCHES = 10

# Factor for reducing the learning rate
__C.TRAIN.GAMMA = 0.1

# Momentum
__C.TRAIN.MOMENTUM = 0.9

# Step size for reducing the learning rate, currently only support one step
__C.TRAIN.STEPSIZE = 10

# Whether to double the learning rate for bias
'''通过这一项就可以让bias的学习率等于对应weight的学习率的2倍'''
__C.TRAIN.DOUBLE_BIAS = True

# '''conv2,conv3,conv4,conv5是否要参与训练'''
# __C.TRAIN.CONV2 = False
# __C.TRAIN.CONV3 = False
# __C.TRAIN.CONV4 = False
# __C.TRAIN.CONV5 = False

# ResNet options
#

__C.RESNET = edict()

# Option to set if max-pooling is appended after crop_and_resize.
# if true, the region will be resized to a square of 2xPOOLING_SIZE,
# then 2x2 max-pooling is applied; otherwise the region will be directly
# resized to a square of POOLING_SIZE
__C.RESNET.MAX_POOL = False

# Number of fixed blocks during training, by default the first of all 4 blocks is fixed
# Range: 0 (none) to 3 (all)
__C.RESNET.FIXED_BLOCKS = 1

# MobileNet options
#

__C.MOBILENET = edict()

# Whether to regularize the depth-wise filters during training
__C.MOBILENET.REGU_DEPTH = False

# Number of fixed layers during training, by default the bottom 5 of 14 layers is fixed
# Range: 0 (none) to 12 (all)
__C.MOBILENET.FIXED_LAYERS = 5

# Weight decay for the mobilenet weights
# __C.MOBILENET.WEIGHT_DECAY = 0.00004

# Depth multiplier
__C.MOBILENET.DEPTH_MULTIPLIER = 1.
# Whether to have weight decay on bias as well
'''不对bias进行decay'''
__C.TRAIN.BIAS_DECAY = False

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[122.6789,116.6688,104.0069]]])

# For reproducibility
# __C.RNG_SEED = 3
# 图片尺寸
__C.TRAIN.IMAGESZ = 128
# Size of the pooled region after RoI pooling
# '''下面是设定的pooling完之后的特征图尺寸'''
# __C.POOLING_SIZE = 7


