# coding=utf-8
import os
import os.path as osp
import tensorflow as tf
import numpy as np
'''2019.2.21 用来测试训练期间的模型是否能融合到tracker中去'''
########################################################################################################################
#                       初始化: 搭建verification network计算图(恢复预训练权重),返回V模型的接口,
########################################################################################################################
'''模型初始化'''
def V_initialize(sess):
    ########## 用VID数据集训练得到的SINT模型的目录
    # V_model_dir = prefix+'/Lasot/SINT_LASOT'
    V_model_dir = '/home/space/Documents/Experiment/ICCV19/Verificator/ckpt'
    '''这个.meta文件保存的仅仅是图结构,而不是实际的参数,改这个东西并不能够选择使用的模型参数'''
    META_DIR = osp.join(V_model_dir, 'SINT_LASOT-65625.meta')
    CHECKPOINT_DIR = V_model_dir
    ########## 使用会话加载之前训练好的模型
    saver = tf.train.import_meta_graph(META_DIR)
    '''真正决定使用的是哪个模型的地方在下面这里.通过修改checkpoint文件的第一行可以选定使用哪个模型参数文件'''
    saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_DIR))
    ########## 得到SINT模型的接口
    # print([n.name for n in tf.get_default_graph().as_graph_def().node])
    # for n in tf.get_default_graph().as_graph_def().node:
    #     if 'embedding' in n.name:
    #         print(n.name)
    graph = tf.get_default_graph()
    template_image_op = graph.get_tensor_by_name('A:0')
    candidates_image_op = graph.get_tensor_by_name('P:0')
    template_feat_op = graph.get_tensor_by_name('GPU_0/embedding:0')
    candidates_feats_op = graph.get_tensor_by_name('GPU_0/embedding_1:0')
    # for v in tf.global_variables():
    #     print(v.name)
    # print(template_image_op)
    # print(template_feat_op)
    # print(candidates_image_op)
    # print(candidates_feats_op)

    return template_image_op,candidates_image_op,template_feat_op,candidates_feats_op

'''实验表明: allow_soft_placement很重要! 因为我的SINT模型是多卡训练的,里面有的tensor名字带GPU1....,在单卡电脑上找不到这种设备就会报错'''
# tfconfig = tf.ConfigProto(allow_soft_placement=True)
# tfconfig.gpu_options.allow_growth = True #allow_growth是防止tensorflow自动占满所有可用显存
# with tf.Session(config=tfconfig) as sess:
#     V_initialize(sess)


