# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import aher_anet
from aher_anet import aher_multibox_adv_layer
import tf_utils
from datetime import datetime
import time
import data_loader
import tf_extended as tfe
import os
import sys
import pandas as pd
from multiprocessing import Process,Queue,JoinableQueue
import multiprocessing
import math
import random
from tf_utils import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('dis_weights', 0.1, 'The weight for the discriminator.')
tf.app.flags.DEFINE_float('gen_weights', 0.1, 'The weight for the generator.')

class Config(object):
    def __init__(self):
        self.learning_rates=[0.001]*100+[0.0001]*100
        #self.training_epochs = len(self.learning_rates)
        self.training_epochs = 1
        self.total_batch_num = 15000
        self.n_inputs = 2048
        self.batch_size = 16
        self.input_steps=512
        self.input_moment_steps=256
        self.gt_hold_num = 25
        self.gt_hold_num_th = 25
        self.batch_size_val=1

# generate context feature
def Context_Train(m_feature,ratio_id,pos_id):
    """ Model and loss function of context information network
        input: m_feature: batch_size x 256 x 2048
        input: position: batch_size 
        output: concate_feature: batch_size x 512 x 2048
    """ 
    config = Config()

    # The start context generator
    net1_i=tf.contrib.layers.conv1d(inputs=m_feature[:,:,],num_outputs=1024,kernel_size=3, \
    stride=1,padding='same',scope='g_conv_s1')

    net1=tf.contrib.layers.conv1d(inputs=net1_i,num_outputs=2048,kernel_size=3, \
    stride=2,padding='same',scope='g_conv_s2')

    # The end context generator
    net2_i=tf.contrib.layers.conv1d(inputs=m_feature[:,:,],num_outputs=1024,kernel_size=3,
    stride=1,padding='same',scope='g_conv_e1')

    net2=tf.contrib.layers.conv1d(inputs=net2_i,num_outputs=2048,kernel_size=3, \
    stride=2,padding='same',scope='g_conv_e2')


    # random crop and select temporal gt
    net_res = []
    temporal_gt = []
    for i in range(config.batch_size):
        ratio = tf.cast(ratio_id[i],tf.float32) * tf.constant(0.05)
        posi = tf.cast(pos_id[i],tf.float32) * tf.constant(0.05)

        resize_fea_len = tf.cast(tf.constant(512.0)*ratio, tf.int32)
        temp_feature = tf.expand_dims(m_feature,2)
        resize_fea = tf.image.resize_images(temp_feature,[resize_fea_len,1])
        reduce_fea = tf.squeeze(resize_fea,2)
        
        net1_len = tf.cast((tf.constant(512)-resize_fea_len)/tf.constant(2),tf.int32)
        net2_len = tf.constant(512)-resize_fea_len-net1_len

        temp_net1 = tf.expand_dims(net1,2)
        resize_net1 = tf.image.resize_images(temp_net1,[net1_len,1])
        inc_net1 = tf.squeeze(resize_net1,2) 

        temp_net2 = tf.expand_dims(net2,2)
        resize_net2 = tf.image.resize_images(temp_net2,[net2_len,1])
        inc_net2 = tf.squeeze(resize_net2,2)
        
        if i % 2 == 0:
            start = tf.cast((tf.constant(1.0)-ratio) * tf.cast(net1_len,tf.float32) * posi, tf.int32)
            net_a = inc_net1[i,:start,]
            net_b = inc_net1[i,start:,]
            net_3 = tf.keras.layers.concatenate(inputs=[net_a,reduce_fea[i,:,],net_b,inc_net2[i,:,]],axis=0)
            net_res.append(tf.reshape(net_3,[1,config.input_steps,config.n_inputs]))
            temporal_gt.append(tf.reshape(tf.cast(start,tf.float32),[1]))
            temporal_gt.append(tf.reshape(tf.cast(start + resize_fea_len,tf.float32),[1]))
        else:
            start = tf.cast((tf.constant(1.0)-ratio) * tf.cast(net2_len,tf.float32) * posi, tf.int32)
            net_a = inc_net2[i,:start,]
            net_b = inc_net2[i,start:,]
            net_3 = tf.keras.layers.concatenate(inputs=[inc_net1[i,:,],net_a,reduce_fea[i,:,],net_b],axis=0)
            net_res.append(tf.reshape(net_3,[1,config.input_steps,config.n_inputs]))
            temporal_gt.append(tf.reshape(tf.cast(start + net1_len,tf.float32),[1]))
            temporal_gt.append(tf.reshape(tf.cast(start + resize_fea_len,tf.float32),[1]))                

    net_c = tf.concat(net_res,axis=0)
    temp_gt = tf.concat(temporal_gt,axis=0)
    temp_gt = tf.reshape(temp_gt,[config.batch_size,1,2])
    
    return net_c,temp_gt

# Discriminator in each anchor layer
def Context_Back_Discriminator(input_points,
        feat_layers=aher_anet.AHERNet.default_params.feat_layers,
        anchor_sizes=aher_anet.AHERNet.default_params.anchor_sizes,
        anchor_ratios=aher_anet.AHERNet.default_params.anchor_ratios,
        normalizations=aher_anet.AHERNet.default_params.normalizations,
        reuse = None):
    
    num_classes = 2
    D_logits = []
    D = []
    for i, layer in enumerate(feat_layers):
        with tf.variable_scope(layer + '_adv',reuse=reuse):
            adv_logits = aher_multibox_adv_layer(input_points[layer],
                                      num_classes,
                                      anchor_sizes[i],
                                      anchor_ratios[i],
                                      normalizations[i])
        D_logits.append(adv_logits)
        D.append(tf.math.sigmoid(adv_logits))
    return D,D_logits

# Background Discriminator
def Adversary_Back_Train(D, D_logits, D_, D_logits_, 
    gscore_untrim, gscore_gene,scope=None):

    with tf.name_scope(scope,'aher_adv_losses'):
        lshape = tfe.get_shape(D_logits[0], 8)
        batch_size = lshape[0]
        fgscore_untrim = []
        fgscore_gene = []
        f_D_logits = []
        f_D_logits_ = []
        f_D = []
        f_D_ = []
        for i in range(len(D_logits)):
            fgscore_untrim.append(tf.reshape(gscore_untrim[i],[-1]))
            fgscore_gene.append(tf.reshape(gscore_gene[i],[-1]))
            f_D_logits.append(tf.reshape(D_logits[i],[-1]))
            f_D_logits_.append(tf.reshape(D_logits_[i],[-1]))
            f_D.append(tf.reshape(D[i],[-1]))
            f_D_.append(tf.reshape(D_[i],[-1]))
        gscore_untrim = tf.concat(fgscore_untrim, axis=0)
        gscore_gene = tf.concat(fgscore_gene, axis=0)
        D_logits = tf.concat(f_D_logits, axis=0)
        D_logits_ = tf.concat(f_D_logits_, axis=0)
        D = tf.concat(f_D, axis=0)
        D_ = tf.concat(f_D_, axis=0)
        dtype = D_logits.dtype

        # select the background position and logits
        pos_mask_untrim = gscore_untrim > 0.70
        nmask_untrim = tf.logical_and(tf.logical_not(pos_mask_untrim),gscore_untrim < 0.3)

        pos_mask_gene = gscore_gene > 0.70
        nmask_gene = tf.logical_and(tf.logical_not(pos_mask_gene),gscore_gene < 0.3)

        nmask = tf.logical_and(nmask_untrim,nmask_gene)
        fnmask = tf.cast(nmask, dtype)
        fnmask_num = tf.reduce_sum(fnmask)

        # compute the sigmoid cross entropy loss
        d_loss_real=sigmoid_cross_entropy_with_logits(D_logits, tf.ones_like(D))
        d_loss_real=tf.div(tf.reduce_sum(d_loss_real*fnmask), fnmask_num/FLAGS.dis_weights, name='d_loss_real')
        
        d_loss_fake=sigmoid_cross_entropy_with_logits(D_logits_, tf.zeros_like(D_))
        d_loss_fake=tf.div(tf.reduce_sum(d_loss_fake*fnmask), fnmask_num/FLAGS.dis_weights, name='d_loss_fake')

        g_loss=sigmoid_cross_entropy_with_logits(D_logits_, tf.ones_like(D_))
        g_loss=tf.div(tf.reduce_sum(g_loss*fnmask), fnmask_num/FLAGS.gen_weights, name='g_loss')     

    return d_loss_real,d_loss_fake,g_loss
