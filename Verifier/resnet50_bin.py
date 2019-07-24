
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys


from nets_bin import resnet_utils_bin
from nets_bin import resnet_v1_bin
from nets_bin.resnet_v1_bin import resnet_v1_block

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.python import pywrap_tensorflow
from Verifier.config import cfg



def resnet_arg_scope(is_training=True,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  batch_norm_params = {
    'is_training': False,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'trainable': False,
    'updates_collections': tf.GraphKeys.UPDATE_OPS
  }

  with arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
      weights_initializer=slim.variance_scaling_initializer(),
      trainable=is_training,
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
      return arg_sc

_num_layers = 50
_scope = 'resnet_v1_%d' % _num_layers
_variables_to_fix = {}
pretrained_model = ''

_blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=1),
        resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
        # use stride 1 for the last conv4 layer
        resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
        resnet_v1_block('block4', base_depth=512, num_units=3, stride=2)]


def _build_base(image,reuse):
  with tf.variable_scope(_scope, _scope):
        net = resnet_utils_bin.conv2d_same(image, 64, 7, stride=2, scope='conv1',reuse=reuse)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
        net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')
  return net

def _image_to_feat(image, is_training, reuse=None):
    assert (0 <= cfg.RESNET.FIXED_BLOCKS <= 3)
    # Now the base is always fixed during training
    with slim.arg_scope(resnet_arg_scope(is_training=False)):
      base_feat = _build_base(image,reuse)
    ''' Residual Block1'''
    with slim.arg_scope(resnet_arg_scope(is_training= False)):
        res1, _ = resnet_v1_bin.resnet_v1(base_feat,
                                          _blocks[0:1],
                                          global_pool=False,
                                          include_root_block=False,
                                          reuse=reuse,
                                          scope=_scope)
        print(res1)
    '''Residual Block2'''
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        res2, _ = resnet_v1_bin.resnet_v1(res1,
                                          _blocks[1:2],
                                          global_pool=False,
                                          include_root_block=False,
                                          reuse=reuse,
                                          scope=_scope)
        print(res2)
    average_pool1 = tf.reduce_mean(res2, axis=[1, 2],name = 'global_average_pool1')
    '''Residual Block3'''
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        res3, _ = resnet_v1_bin.resnet_v1(res2,
                                          _blocks[2:3],
                                          global_pool=False,
                                          include_root_block=False,
                                          reuse=reuse,
                                          scope=_scope)
        print(res3)
    average_pool2 = tf.reduce_mean(res3, axis=[1, 2],name = 'global_average_pool2')
    '''Residual Block4'''
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        res4, _ = resnet_v1_bin.resnet_v1(res3,
                                          _blocks[3:],
                                          global_pool=False,
                                          include_root_block=False,
                                          reuse=reuse,
                                          scope=_scope)
        print(res4)
    average_pool3 = tf.reduce_mean(res4, axis=[1, 2], name='global_average_pool3')
    bottleneck = tf.concat([average_pool1,average_pool2,average_pool3],axis=1,name='concat')
    print(bottleneck)
    if is_training:
        bottleneck = slim.dropout(bottleneck, keep_prob=0.5, is_training=True, scope='dropout')
    unnormal_embedding = slim.fully_connected(bottleneck, 128, scope='fc',activation_fn = None, reuse = reuse, weights_regularizer=slim.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))

    embedding = tf.nn.l2_normalize(unnormal_embedding,axis=-1)
    final_embedding = tf.identity(embedding,name='embedding')
    # print(embedding)

    return final_embedding
def get_variables_to_restore(variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
        # exclude the first conv layer to swap RGB to BGR
        if v.name == (_scope + '/conv1/weights:0'):
            _variables_to_fix[v.name] = v
            continue
        if v.name.split(':')[0] in var_keep_dic:
            print('Variables restored: %s' % v.name)
            variables_to_restore.append(v)

    return variables_to_restore
def get_variables_in_checkpoint_file(file_name):
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")

def fix_variables(sess, pretrained_model):
    print('Fix Resnet V1 layers..')
    with tf.variable_scope('Fix_Resnet_V1') as scope:
        with tf.device("/cpu:0"):
            # fix RGB to BGR
            conv1_rgb = tf.get_variable("conv1_rgb", [7, 7, 3, 64], trainable=False)
            restorer_fc = tf.train.Saver({_scope + "/conv1/weights": conv1_rgb})
            restorer_fc.restore(sess, pretrained_model)

            sess.run(tf.assign(_variables_to_fix[_scope + '/conv1/weights:0'],
                               tf.reverse(conv1_rgb, [2])))
def restore_variables(sess):

    # Fresh train directly from ImageNet weights
    print('Loading initial model weights from {:s}'.format(pretrained_model))

    variables = tf.global_variables() # list
    # Initialize all variables first
    sess.run(tf.variables_initializer(variables, name='init'))
    var_keep_dic = get_variables_in_checkpoint_file(pretrained_model)
    # Get the variables to restore, ignoring the variables to fix
    variables_to_restore = get_variables_to_restore(variables, var_keep_dic)

    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, pretrained_model)
    print('Loaded.')
    # Need to fix the variables before loading, so that the RGB weights are changed to BGR
    # For VGG16 it also changes the convolutional weights fc6 and fc7 to
    # fully connected weights
    fix_variables(sess, pretrained_model)
    print('Fixed.')
def _create_Siamese_network(A,P,N,NOT_FISRT_CLONE):
    if NOT_FISRT_CLONE:
        featA = _image_to_feat(A, is_training=True, reuse=True)
        featP = _image_to_feat(P, is_training=True, reuse=True)
        featN = _image_to_feat(N, is_training=True, reuse=True)
    else:
        featA = _image_to_feat(A, is_training=True, reuse=False)
        featP = _image_to_feat(P, is_training=True, reuse=True)
        featN = _image_to_feat(N, is_training=True, reuse=True)

    return featA,featP,featN


