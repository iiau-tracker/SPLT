
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import _init_paths # add additional path
from config import cfg
from parse_tfrecords import get_inputs
from resnet50_bin import _create_Siamese_network,restore_variables
cfg.TRAIN.NUM_GPUS = 2
cfg.TRAIN.LEARNING_RATE = 1e-2 # Bigger batch ---> Bigger learning rate
cfg.TRAIN.WEIGHT_DECAY = 5e-4
cfg.TRAIN.MARGIN = 0.2 # as same as FaceNet
cfg.TRAIN.EPOCHES = 70
cfg.TRAIN.BATCH_SIZE = 32
'''remember to change options below ! ! !'''
SUMMARY_PATH = './summary_resnet50_VID_N_cascade'
checkpoint_path = "./V_resnet50_VID_N_cascade/V_resnet50_VID_N_cascade"
'''Triplet Loss'''
def get_loss(A,P,N,NOT_FIRST_CLONE):
    featA,featP,featN = _create_Siamese_network(A,P,N,NOT_FIRST_CLONE)
    with tf.name_scope("losses"):
        pos_dist = tf.reduce_sum(tf.square(featA - featP), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(featA - featN), axis=-1)
        triplet_loss = tf.reduce_mean(tf.maximum(pos_dist - neg_dist + cfg.TRAIN.MARGIN,0))
        regula_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
        loss = triplet_loss + regula_loss
        tf.summary.scalar('triplet_loss',triplet_loss)
        tf.summary.scalar('regularization_loss',regula_loss)
        tf.summary.scalar('total_loss',loss)
    return loss
def average_gradients(tower_grads):
    average_grads = []

    for grads_and_vars in zip(*tower_grads):

        grads = []
        for g, _ in grads_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grads_and_vars[0][1]
        gard_and_var = (grad, v)

        average_grads.append(gard_and_var)
    return average_grads
def main(argv=None):
    with tf.Graph().as_default(),tf.device('/cpu:0'):
        A,P,N = get_inputs()
        A_input = tf.identity(A, name='A')
        P_input = tf.identity(P, name='P')
        N_input = tf.identity(N, name='N')
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE,global_step,
                                        (60000*20)/(cfg.TRAIN.BATCH_SIZE*cfg.TRAIN.NUM_GPUS),1/10)
        tf.summary.scalar('learning_rate',lr)
        opti = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        tower_grads = []
        NOT_FIRST_CLONE = False

        for gpu in range(cfg.TRAIN.NUM_GPUS):
            with tf.device('/gpu:%d' % gpu):
                with tf.name_scope('GPU_%d' % gpu) as scope:

                    cur_loss = get_loss(A_input,P_input,N_input,NOT_FIRST_CLONE)
                    NOT_FIRST_CLONE = True
                    grads = opti.compute_gradients(cur_loss)
                    tower_grads.append(grads)
        '''everything to visualize'''
        summary_op = tf.summary.merge_all()
        # compute average gradient
        grads = average_gradients(tower_grads)
        apply_gradient_op = opti.apply_gradients(grads,global_step=global_step)
        saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement = True)) as sess:
            init.run()
            summary_writer = tf.summary.FileWriter(SUMMARY_PATH, sess.graph)

            restore_variables(sess)
            TRAINING_STEPS = int(cfg.TRAIN.EPOCHES * 60000 /(cfg.TRAIN.NUM_GPUS*cfg.TRAIN.BATCH_SIZE))
            starttime = time.time()
            for step in range(TRAINING_STEPS):
                _, summary = sess.run([apply_gradient_op, summary_op])
                # loss_list,_,summary = sess.run([loss,apply_gradient_op,summary_op])
                print('step%d done'%step)
                summary_writer.add_summary(summary, step)

                if step % (int(60000*10/cfg.TRAIN.NUM_GPUS/cfg.TRAIN.BATCH_SIZE)) == 0 or (step+1) == TRAINING_STEPS:
                    saver.save(sess, checkpoint_path, global_step = step, write_meta_graph=True)
        endtime = time.time()
        T = endtime - starttime
        print('Total time consumed is %4f hours'%(T/3600))

if __name__ == '__main__':
    tf.app.run()

