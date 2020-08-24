# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import aher_anet
import tf_utils
from datetime import datetime
import time
from data_loader import *
from aher_adv import *
from tf_utils import *
import tf_extended as tfe
import os
import sys
import pandas as pd
from multiprocessing import Process,Queue,JoinableQueue
import multiprocessing
import math
import random


FLAGS = tf.app.flags.FLAGS
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

# Flags for adversary training
tf.app.flags.DEFINE_float('beta1', 0.5, 'Momentum term of adam [0.5]')
tf.app.flags.DEFINE_float('param_adv_learning_rate', 0.0001, 'Init learning rate for adversary learning')
tf.app.flags.DEFINE_float('param_initial_loc_learning_rate', 0.0001, 'Init learning rate for localization learning')
# Flags for localization training
tf.app.flags.DEFINE_float('loss_alpha', 1., 'Alpha parameter in the loss function.')
tf.app.flags.DEFINE_float('match_threshold', 0.80, 'Matching threshold in the loss function.')
tf.app.flags.DEFINE_float('neg_match_threshold', 0.30, 'Negative threshold in the loss function.')
tf.app.flags.DEFINE_float('negative_ratio', 1., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float('label_smoothing', 0.0, 'The amount of label smoothing.')
# Flags for weights
tf.app.flags.DEFINE_float('cls_weights', 2.0, 'The weight for the classification.')
tf.app.flags.DEFINE_float('iou_weights', 25.0, 'The weight for the iou prediction.')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001,'Initial learning rate.')
# Flags for box generation
tf.app.flags.DEFINE_float('select_threshold', 0.0, 'Selection threshold.')
tf.app.flags.DEFINE_float('nms_threshold', 0.90, 'Non-Maximum Selection threshold.')
tf.app.flags.DEFINE_integer('select_top_k', 765, 'Select top-k detected bounding boxes.')
tf.app.flags.DEFINE_integer('keep_top_k', 100, 'Keep top-k detected objects.')
tf.app.flags.DEFINE_bool('cls_flag', True, 'Utilize classification score.')

def AHER_init():
    # AHER parameter and Model
    aher_anet_param = aher_anet.AHERNet.default_params
    aher_anet_model = aher_anet.AHERNet(aher_anet_param)
    aher_anet_temporal_shape = aher_anet_param.temporal_shape
    aher_anet_anchor = aher_anet_model.anchors(aher_anet_temporal_shape)

    return aher_anet_model,aher_anet_anchor

def AHER_Predictor_Cls(aher_anet_model,aher_anet_anchor,
    feature,temporal_gt,vname,label,duration,reuse,class_num=600,cls_suffix='_anchor'):
    """ Model and loss function of sigle shot action localization
        feature:      batch_size x 512 x 2048
        temporal_gt:  batch_size x 25 x 2
        vname:        batch_size x 1
        label:        batch_size x 1
        duration:     batch_size x 1
    """ 

    # Encode groundtruth labels and bboxes.
    gclasses, glocalisations, gscores, giou = \
        aher_anet_model.bboxes_encode(label, temporal_gt, aher_anet_anchor)

    # predict location and iou
    predictions, localisation, logits, proplogits, proppredictions, iouprediction, clsweights, clsbias, end_points  = \
        aher_anet_model.net_pool_cls(feature, num_classes= class_num, untrim_num=class_num, is_training=True,reuse=reuse,cls_suffix=cls_suffix)

    return predictions, localisation, logits, proplogits, iouprediction, \
    clsweights, clsbias, gscores, giou, gclasses, glocalisations, end_points

def AHER_Predictor_Prop(aher_anet_model,aher_anet_anchor,
    feature,temporal_gt,vname,label,duration,reuse):
    """ Model and loss function of sigle shot action localization
        feature:      batch_size x 512 x 4069
        temporal_gt:  batch_size x 25 x 2
        vname:        batch_size x 1
        label:        batch_size x 1
        duration:     batch_size x 1
    """ 

    # Encode groundtruth labels and bboxes.
    gclasses, glocalisations, gscores, giou = \
        aher_anet_model.bboxes_encode(label, temporal_gt, aher_anet_anchor)

    # predict location and proppredictions
    predictions, localisation, logits, proplogits, proppredictions, iouprediction, end_points \
       = aher_anet_model.net_prop_iou_pure(feature,is_training=True,reuse=reuse,cls_suffix='_anet')


    return predictions, localisation, logits, proplogits, iouprediction, gscores, giou, \
    gclasses, glocalisations, end_points

def AHER_Predictor_Weights_Prop(aher_anet_model,aher_anet_anchor,
    feature,temporal_gt,vname,label,duration,clsweights,clsbias,reuse,n_class,cls_suffix='_anet'):
    """ Model and loss function of sigle shot action localization
        feature:      batch_size x 512 x 4069
        temporal_gt:  batch_size x 25 x 2
        vname:        batch_size x 1
        label:        batch_size x 1
        duration:     batch_size x 1
    """ 

    # Encode groundtruth labels and bboxes.
    gclasses, glocalisations, gscores, giou = \
        aher_anet_model.bboxes_encode(label, temporal_gt, aher_anet_anchor)

    # predict location and proppredictions
    predictions, localisation, logits, proplogits, proppredictions, iouprediction, end_points \
       = aher_anet_model.net_prop_iou(feature,clsweights,clsbias,is_training=True,reuse=reuse,num_classes=n_class,cls_suffix=cls_suffix)


    return predictions, localisation, logits, proplogits, iouprediction, gscores, giou, \
    gclasses, glocalisations, end_points

def AHER_Inference(aher_anet_model,aher_anet_anchor,
                   feature,vname,label,duration,clsweights,reuse):
    """ Inference bbox of sigle shot action localization
        feature:      batch_size x 512 x 4069
        vname:        batch_size x 1
        label:        batch_size x 1
        duration:     batch_size x 1
    """ 
    # predict location and iou
    predictions, localisation, logits, proplogits, proppredictions, end_points = \
        aher_anet_model.net_prop(feature,clsweights, is_training=False,reuse=reuse)

    # decode bounding box and get scores
    localisation = aher_anet_model.bboxes_decode(localisation, duration ,aher_anet_anchor)
    if FLAGS.cls_flag:
        rscores, rbboxes = aher_anet_model.detected_bboxes(proppredictions, localisation,                                        
                                    select_threshold=FLAGS.select_threshold,
                                    nms_threshold=FLAGS.nms_threshold,
                                    clipping_bbox=None,
                                    top_k=FLAGS.select_top_k,
                                    keep_top_k=FLAGS.keep_top_k,
                                    iou_flag=False)
    else:
        rscores, rbboxes = aher_anet_model.detected_bboxes(proppredictions, localisation,                                        
                                    select_threshold=FLAGS.select_threshold,
                                    nms_threshold=FLAGS.nms_threshold,
                                    clipping_bbox=None,
                                    top_k=FLAGS.select_top_k,
                                    keep_top_k=FLAGS.keep_top_k,
                                    iou_flag=True)
       
    prebbox={"rscores":rscores,"rbboxes":rbboxes}
    return prebbox

def AHER_Train(aher_anet_model,logits,localisation,proplogits, iouprediction,
              gclasses, glocalisations, gscores, giou, AHER_trainable_variables, LR):
    # loss function
    loss_prop,loss_loc,loss_cls,acc_cls,loss_iou = \
    aher_anet_model.losses_complete(logits,localisation,proplogits,iouprediction,
                           gclasses, glocalisations, gscores, giou,
                           match_threshold=FLAGS.match_threshold,
                           neg_match_threshold=FLAGS.neg_match_threshold,
                           negative_ratio=FLAGS.negative_ratio,
                           alpha=FLAGS.loss_alpha,
                           label_smoothing=FLAGS.label_smoothing,
                           cls_weights=FLAGS.cls_weights,
                           iou_weights=FLAGS.iou_weights)

    total_loss = loss_prop + loss_loc + loss_cls + loss_iou
    proposal_loss = loss_prop + loss_loc + loss_iou

    loss={"Total_loss":total_loss, "Prop_loss":proposal_loss, "loss_prop":loss_prop,
          "loss_loc":loss_loc, "loss_iou":loss_iou ,"loss_cls":loss_cls,"acc_cls":acc_cls}

    # first to train the proposal net
    optimizer_prop=tf.train.AdamOptimizer(learning_rate=LR).minimize(proposal_loss,var_list=AHER_trainable_variables)
    optimizer_cls=tf.train.AdamOptimizer(learning_rate=LR).minimize(loss_cls,var_list=AHER_trainable_variables)
    
    optimizer = tf.group(optimizer_prop,optimizer_cls)

    return optimizer,loss

def AHER_Cls_Train(aher_anet_model, logits, gclasses, gscores, AHER_trainable_variables, LR):
    # loss function
    loss_cls,acc_cls = \
    aher_anet_model.reg_losses(logits,
                           gclasses,
                           match_threshold=FLAGS.match_threshold,
                           negative_ratio=FLAGS.negative_ratio,
                           alpha=FLAGS.loss_alpha,
                           label_smoothing=FLAGS.label_smoothing,
                           cls_weights=FLAGS.cls_weights,
                           iou_weights=FLAGS.iou_weights)

    total_loss = loss_cls
    classification_loss = loss_cls

    loss={"loss_cls":loss_cls,"acc_cls":acc_cls}

    optimizer=tf.train.AdamOptimizer(learning_rate=LR).minimize(classification_loss,var_list=AHER_trainable_variables)

    return optimizer,loss

class Config(object):
    def __init__(self):
        self.learning_rates=[0.001]*100+[0.0001]*100
        self.training_epochs = 1
        self.total_batch_num = 15000
        self.n_inputs = 2048
        self.batch_size = 16
        self.input_steps=512
        self.input_moment_steps=256
        self.gt_hold_num = 25
        self.gt_hold_num_th = 25
        self.batch_size_val=1

def get_learning_rate(global_iternum, boundaries, values):
    lr = values[0]
    for m in range(len(boundaries)-1):
        if m == 0 and global_iternum <= boundaries[0]:
            return values[0]
        if global_iternum > boundaries[m] and global_iternum <= boundaries[m+1]:
            return values[m+1] 
    if global_iternum > boundaries[-1]:
        return values[-1]
    else: return lr

if __name__ == "__main__":

    snapshot_step = 1000

    csv_dir = ''
    csv_oris_dir = '/data/Kinetics-600/csv_p3d_clip_trim/' # original clip feature for Kinetics-600
    csv_dir_anet = '/data/ActivityNet/csv_mean_512_p3d_clip_2018' # resized clip feature for ActivityNet
    csv_oris_dir_anet= '/data/ActivityNet/csv_p3d_clip' # original clip feature for ActivityNet 

    model_output_dir = 'p3d_k600_models/AHER_k600_gen_adv'
    print('Output folder name: %s'%(model_output_dir))
    """ define the input and the network""" 
    config = Config()
    LR= tf.placeholder(tf.float32)




    #---------------------------------------------------- Variable Placeholder ----------------------------------------------------------# 
    # untrim video (ActivityNet)
    feature = tf.placeholder(tf.float32, shape=(config.batch_size,config.input_steps,config.n_inputs))
    temporal_gt = tf.placeholder(tf.float32, shape=(config.batch_size, config.gt_hold_num_th, 2))
    vname = tf.placeholder(tf.string, shape=(config.batch_size))
    label = tf.placeholder(tf.int32, shape=(config.batch_size))
    duration = tf.placeholder(tf.float32,shape=(config.batch_size))

    # untrim video foreground segment for classification (ActivityNet foreground)
    feature_fore = tf.placeholder(tf.float32, shape=(config.batch_size,config.input_steps,config.n_inputs))
    temporal_gt_fore = tf.placeholder(tf.float32, shape=(config.batch_size, 1, 2))
    vname_fore = tf.placeholder(tf.string, shape=(config.batch_size))
    label_fore = tf.placeholder(tf.int32, shape=(config.batch_size))
    duration_fore = tf.placeholder(tf.float32,shape=(config.batch_size))    

    # moments video for classification (Kinetics)
    feature_seg = tf.placeholder(tf.float32, shape=(config.batch_size,config.input_moment_steps,config.n_inputs))
    temporal_gt_seg = tf.placeholder(tf.float32, shape=(config.batch_size, 1, 2))
    vname_seg = tf.placeholder(tf.string, shape=(config.batch_size))
    label_seg = tf.placeholder(tf.int32, shape=(config.batch_size))
    duration_seg = tf.placeholder(tf.float32,shape=(config.batch_size))    
    ratio_seg = tf.placeholder(tf.int32,shape=(config.batch_size))
    position_seg = tf.placeholder(tf.int32,shape=(config.batch_size))
    #--------------------------------------------------------------------------------------------------------------------------------------# 




    #---------------------------------------------------- AherNet Structure ----------------------------------------------------------------# 
    # init AHER net
    aher_anet_model,aher_anet_anchor = AHER_init()

    # Initialize the backbone with moments (Kinetics) input
    predictions_mots, localisation_mots, logits_mots, proplogits_mots, iouprediction_mots, \
    clsweights_ki, clsbias_ki, gscore_mots, \
    giou_mots, gclasses_mots, glocalisations_mots, \
    end_points_mots \
    = AHER_Predictor_Cls(aher_anet_model,aher_anet_anchor,feature_seg, \
    temporal_gt_seg,vname_seg,label_seg,duration_seg,reuse=False,class_num=600,cls_suffix='_anchor')    

    # generate action context of moments (Kinetics) and concatenation
    feature_generate_seg,temporal_gt_seg_random = Context_Train(feature_seg,ratio_seg,position_seg)
    # Train on synthetic moments (Kinetics) for localizatin
    predictions_gene_seg, localisation_gene_seg, logits_gene_seg, proplogits_gene_seg, iouprediction_gene_seg, \
    gscore_gene_seg, \
    giou_gene_seg, gclasses_gene_seg, glocalisations_gene_seg, \
    end_points_gene_seg \
    = AHER_Predictor_Weights_Prop(aher_anet_model,aher_anet_anchor,feature_generate_seg, \
    temporal_gt_seg_random,vname_seg,label_seg,duration_seg, clsweights_ki, clsbias_ki,reuse=True,n_class=600,cls_suffix='_anchor')

    # Train on untrimmed foreground segment (ActivityNet) for classificaton
    predictions_gene_fore, localisation_gene_fore, logits_gene_fore, proplogits_gene_fore, iouprediction_gene_fore, \
    clsweights_fore, clsbias_fore, gscore_gene_fore, \
    giou_gene_fore, gclasses_gene_fore, glocalisations_gene_fore, \
    end_points_gene_fore \
    = AHER_Predictor_Cls(aher_anet_model,aher_anet_anchor,feature_fore, \
    temporal_gt_fore,vname_fore,label_fore,duration_fore,reuse=True,class_num=200,cls_suffix='_anet')

    # Train on untrimmed video (ActivityNet) for localization to learn transfer function
    predictions_untrim, \
    localisation_untrim, logits_untrim, \
    proplogits_untrim, iouprediction_untrim, gscore_untrim, \
    giou_untrim, gclasses_untrim, glocalisations_untrim, \
    end_points_untrim \
    = AHER_Predictor_Weights_Prop(aher_anet_model,aher_anet_anchor,feature,temporal_gt, \
    vname,label,duration,clsweights_fore, clsbias_fore, reuse=True,n_class=200,cls_suffix='_anet')

    # adversary loss of Background cell 
    D, D_logits   = Context_Back_Discriminator(end_points_untrim, reuse=False)
    D_, D_logits_   = Context_Back_Discriminator(end_points_gene_seg, reuse=True)
    d_loss_real,d_loss_fake,g_loss = Adversary_Back_Train(D, D_logits, D_, D_logits_, \
    gscore_untrim, gscore_gene_seg)
    d_loss = d_loss_real + d_loss_fake

    loss_adv_dis={"loss_dis":d_loss}
    loss_adv_gen={"loss_gen":g_loss}

    # adversary optimizer
    AHER_trainable_variables = tf.trainable_variables()
    d_vars = []
    for var in AHER_trainable_variables:
        if '_adv' in var.name: d_vars.append(var)
    g_vars = []
    for var in AHER_trainable_variables:
        if 'g_conv_s' in var.name: g_vars.append(var)
        if 'g_conv_e' in var.name: g_vars.append(var)
    loc_vars = [var for var in AHER_trainable_variables if 'aher' in var.name]
    loc_vars_anet_loc_s = [var for var in loc_vars if 'conv_action_anchor' not in var.name]
    loc_vars_anet_loc = [var for var in loc_vars_anet_loc_s if 'conv_action_anet' not in var.name]
    loc_vars_anet_cls = [var for var in AHER_trainable_variables if 'conv_action_anet' in var.name]
    loc_vars_kinetics_s = [var for var in loc_vars if 'conv_action_anet' not in var.name]
    loc_vars_kinetics_s2 = [var for var in loc_vars_kinetics_s if 'fully' not in var.name]
    loc_vars_kinetics = [var for var in loc_vars_kinetics_s2 if 'conv_action_anchor' not in var.name]
    backbone_vars = [var for var in loc_vars_kinetics_s2 if 'box' not in var.name]
    anchor_vars = [var for var in loc_vars_kinetics_s if 'conv_action_anchor' in var.name]
    kinetics_cls_vars = backbone_vars + anchor_vars

    d_optim = tf.train.AdamOptimizer(FLAGS.param_adv_learning_rate, \
              beta1=FLAGS.beta1) \
              .minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(FLAGS.param_adv_learning_rate, \
              beta1=FLAGS.beta1) \
              .minimize(g_loss, var_list=g_vars)
    #--------------------------------------------------------------------------------------------------------------------------------------# 




    #---------------------------------------------------- Training Loss Function -----------------------------------------------------------# 
    # 1. classification on foreground segment of untrimmed video (ActivityNet)
    optimizer_fore_cls,loss_fore_cls = \
        AHER_Cls_Train(aher_anet_model, logits_gene_fore, gclasses_gene_fore, gscore_gene_fore, loc_vars_anet_cls, LR)

    # 2. classification on moments (Kinetics)
    optimizer_ki_cls,loss_ki_cls = \
        AHER_Cls_Train(aher_anet_model, logits_mots, gclasses_mots, gscore_mots, kinetics_cls_vars, LR)

    # 3. localization on untrimmed video (ActivityNet) to learn transfer function
    optimizer_untrim,loss_untrim = \
        AHER_Train(aher_anet_model,logits_untrim,localisation_untrim,proplogits_untrim, iouprediction_untrim, \
        gclasses_untrim, glocalisations_untrim, gscore_untrim, giou_untrim, loc_vars_anet_loc, LR)

    # 4. localization on synthetic moments (Kinetics) to learn generators
    optimizer_seg,loss_seg = \
        AHER_Train(aher_anet_model,logits_gene_seg,localisation_gene_seg,proplogits_gene_seg, iouprediction_gene_seg, \
        gclasses_gene_seg, glocalisations_gene_seg, gscore_gene_seg, giou_gene_seg, g_vars, LR)

    # 5. localization on synthetic moments (Kinetics) to learn 1D backbone
    optimizer_bone,loss_bone = \
        AHER_Train(aher_anet_model,logits_gene_seg,localisation_gene_seg,proplogits_gene_seg, iouprediction_gene_seg, \
        gclasses_gene_seg, glocalisations_gene_seg, gscore_gene_seg, giou_gene_seg, loc_vars_kinetics, LR)    
    #--------------------------------------------------------------------------------------------------------------------------------#




    #---------------------------------------------------- Data Input Stream --------------------------------------------------------# 
    # ActivityNet untrimmed data generator
    print('Build the train data for ActivityNet.')
    data_generate = AnetDatasetLoadMultiProcessQueue('train',csv_dir_anet,25,512,df_file='./cvs_records_anet/anet_train_val_reduce_info.csv')
    dataset_train = tf.data.Dataset.from_generator(data_generate.gen,
    (tf.float32, tf.float32, tf.string, tf.int32, tf.float32),
    (tf.TensorShape([512, 2048]),tf.TensorShape([25, 2]),tf.TensorShape([]),tf.TensorShape([]),tf.TensorShape([])))
    dataset_train = dataset_train.batch(config.batch_size)
    batch_num_train = int(len(data_generate.train_list) / config.batch_size)    
    iterator_train = dataset_train.make_one_shot_iterator()
    feature_g, \
    video_gt_g, \
    video_name_g, \
    video_label_g, \
    video_duration_g  = iterator_train.get_next() 

    # ActivityNet foreground segment data generator
    print('Build the foreground data for ActivityNet.')
    data_fore = AnetDatasetLoadForeQueue('train',csv_dir_anet,csv_oris_dir_anet,25,512,df_file='./cvs_records_anet/anet_train_val_reduce_info.csv',resize_dim=512)
    dataset_fore = tf.data.Dataset.from_generator(data_fore.gen,
    (tf.float32, tf.float32, tf.string, tf.int32, tf.float32),
    (tf.TensorShape([512, 2048]),tf.TensorShape([1, 2]),tf.TensorShape([]),tf.TensorShape([]),tf.TensorShape([])))
    dataset_fore = dataset_fore.batch(config.batch_size)
    batch_num_fore = int(len(data_fore.train_list) / config.batch_size)    
    iterator_fore = dataset_fore.make_one_shot_iterator()
    feature_g_fore, \
    video_gt_g_fore, \
    video_name_g_fore, \
    video_label_g_fore, \
    video_duration_g_fore  = iterator_fore.get_next()    

    # Kinetics moment data generator
    print('Build the train data for Kinetics.')
    data_moment_generate = KineticsDatasetLoadTrimMultiProcess('train',csv_dir,csv_oris_dir,25,512,df_file='./cvs_records_k600/k600_train_val_info.csv',
    resize_dim=256,ratio_sid=14,ratio_eid=19)       
    dataset_moment = tf.data.Dataset.from_generator(data_moment_generate.gen_pos_random,
    (tf.float32, tf.float32, tf.string, tf.int32, tf.float32, tf.int32, tf.int32),
    (tf.TensorShape([256, 2048]),tf.TensorShape([1, 2]),tf.TensorShape([]),tf.TensorShape([]),tf.TensorShape([]),tf.TensorShape([]),tf.TensorShape([])))
    dataset_moment = dataset_moment.batch(config.batch_size)
    batch_num_moment = int(len(data_moment_generate.train_list) / config.batch_size)    
    iterator_moment = dataset_moment.make_one_shot_iterator()
    feature_g_mome, \
    video_gt_g_mome, \
    video_name_g_mome, \
    video_label_g_mome, \
    video_duration_g_mome, \
    video_ratio_g_mome, \
    video_position_g_mome  = iterator_moment.get_next()  
    #--------------------------------------------------------------------------------------------------------------------------------#




    #---------------------------------------------------- Log and Model Initialization ---------------------------------------------# 
    """ Init tf""" 
    model_saver=tf.train.Saver(var_list=AHER_trainable_variables,max_to_keep=80)
    prop_saver = tf.train.Saver(var_list=loc_vars_anet_loc,max_to_keep=80)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement =True
    sess=tf.InteractiveSession(config=tf_config) 

    train_info={"Total_loss":[], "Prop_loss":[],"loss_cls":[],"loss_loc":[],"loss_prop":[],"loss_iou":[],"acc_cls":[]}
    train_info_bone = {"Total_loss":[], "Prop_loss":[],"loss_cls":[],"loss_loc":[],"loss_prop":[],"loss_iou":[],"acc_cls":[]}
    train_adv_dis_info={"loss_dis":[]}
    train_adv_gen_info={"loss_gen":[]}
    val_info={"Total_loss":[], "Prop_loss":[], "loss_cls":[],"loss_loc":[],"loss_prop":[],"acc_cls":[]}

    info_keys=train_info.keys()
    info_adv_dis_keys=train_adv_dis_info.keys()
    info_adv_gen_keys=train_adv_gen_info.keys()
    best_val_cost = 1000000

    boundaries = [15000, 30000, 45000, 60000, 75000]
    values = [FLAGS.param_initial_loc_learning_rate, 
              FLAGS.param_initial_loc_learning_rate*0.1, 
              FLAGS.param_initial_loc_learning_rate*0.01,
              FLAGS.param_initial_loc_learning_rate*0.001, 
              FLAGS.param_initial_loc_learning_rate*0.0001, 
              FLAGS.param_initial_loc_learning_rate*0.00001]
    global_iternum = 0

    total_batch_idx = 0
    fw_log = open('%s/loss_results_log.txt'%(model_output_dir),'w',1)
    #--------------------------------------------------------------------------------------------------------------------------------#




    #---------------------------------------------------- Train AherNet --------------------------------------------------------------# 
    with tf.Session() as sess:
        tf.global_variables_initializer().run()  
        tf.local_variables_initializer().run()
        # Restore the initialization 1D backbone on the ActivityNet data
        # prop_saver.restore(sess,'p3d_anet_models/aher_model_checkpoint-step_95000')
        print('Begin optimizer. batch_num_moment: %d batch_untr: %d'%(batch_num_moment,batch_num_train))
        mini_info={"Total_loss":[], "Prop_loss":[] ,"loss_cls":[],"loss_loc":[],"loss_prop":[],"loss_iou":[],"acc_cls":[]}
        mini_adv_dis_info={"loss_dis":[]}
        mini_adv_gen_info={"loss_gen":[]}
        mini_info_bone={"Total_loss":[], "Prop_loss":[] ,"loss_cls":[],"loss_loc":[],"loss_prop":[],"loss_iou":[],"acc_cls":[]}
        for epoch in range(0,config.training_epochs):
            """ Training""" 
            for idx in range(config.total_batch_num):
                batch_anchor_feature_mome, \
                batch_video_gt_mome, \
                batch_video_name_mome, \
                batch_video_label_mome, \
                batch_video_duration_mome,\
                batch_video_ratio_mome, \
                batch_video_position_mome = sess.run([feature_g_mome,video_gt_g_mome,video_name_g_mome,video_label_g_mome,video_duration_g_mome,video_ratio_g_mome,video_position_g_mome])

                batch_anchor_feature, \
                batch_video_gt, \
                batch_video_name, \
                batch_video_label, \
                batch_video_duration = sess.run([feature_g,video_gt_g,video_name_g,video_label_g,video_duration_g])          

                batch_anchor_feature_fore, \
                batch_video_gt_fore, \
                batch_video_name_fore, \
                batch_video_label_fore, \
                batch_video_duration_fore = sess.run([feature_g_fore,video_gt_g_fore,video_name_g_fore,video_label_g_fore,video_duration_g_fore])        

                lr_now = get_learning_rate(global_iternum, boundaries, values)

                # --------------------------------------   AherNet Optimization ---------------------------------------------------------#
                # 1. optimize the classification on moments (Kinetics)
                _,out_loss_ki=sess.run([optimizer_ki_cls,loss_ki_cls], 
                                 feed_dict={feature_seg:batch_anchor_feature_mome,
                                            temporal_gt_seg:batch_video_gt_mome,
                                            vname_seg:batch_video_name_mome,
                                            label_seg:batch_video_label_mome,
                                            duration_seg:batch_video_duration_mome,
                                            LR:lr_now})       
                
                # 2. optimize the classification on untrimmed video foreground (ActivityNet)
                _,out_loss_fore=sess.run([optimizer_fore_cls,loss_fore_cls], 
                                 feed_dict={feature_fore:batch_anchor_feature_fore,
                                            temporal_gt_fore:batch_video_gt_fore,
                                            vname_fore:batch_video_name_fore,
                                            label_fore:batch_video_label_fore,
                                            duration_fore:batch_video_duration_fore,
                                            LR:lr_now})
                
                # 3. optimize the localization on untrimmed videos (ActivityNet) to learn transfer function
                _,out_loss_anet=sess.run([optimizer_untrim,loss_untrim], 
                                 feed_dict={feature:batch_anchor_feature,
                                            temporal_gt:batch_video_gt,
                                            vname:batch_video_name,
                                            label:batch_video_label,
                                            duration:batch_video_duration,
                                            LR:lr_now}) 
                
                # 4. train localization model on synthetic moments (Kinetics) to optmize generators
                _,out_loss_untrim=sess.run([optimizer_seg,loss_seg], 
                                 feed_dict={feature_seg:batch_anchor_feature_mome,
                                            temporal_gt_seg:batch_video_gt_mome,
                                            vname_seg:batch_video_name_mome,
                                            label_seg:batch_video_label_mome,
                                            duration_seg:batch_video_duration_mome,
                                            ratio_seg: batch_video_ratio_mome,
                                            position_seg:batch_video_position_mome,
                                            LR:lr_now})  
                
                # 5. optimize the adversary loss of background cells
                #------ Update D network
                _, out_d_loss = sess.run([d_optim, loss_adv_dis],
                  feed_dict={feature:batch_anchor_feature,
                             temporal_gt:batch_video_gt,
                             vname:batch_video_name,
                             label:batch_video_label,
                             duration:batch_video_duration,
                             feature_seg:batch_anchor_feature_mome,
                             temporal_gt_seg:batch_video_gt_mome,
                             vname_seg:batch_video_name_mome,
                             label_seg:batch_video_label_mome,
                             duration_seg:batch_video_duration_mome,
                             ratio_seg: batch_video_ratio_mome,
                             position_seg:batch_video_position_mome})
                #------ Update G network
                _, out_g_loss1 = sess.run([g_optim, loss_adv_gen],
                  feed_dict={feature_seg:batch_anchor_feature_mome,
                            temporal_gt_seg:batch_video_gt_mome,
                            vname_seg:batch_video_name_mome,
                            label_seg:batch_video_label_mome,
                            duration_seg:batch_video_duration_mome,
                            ratio_seg: batch_video_ratio_mome,
                            position_seg:batch_video_position_mome,
                            feature:batch_anchor_feature,
                            temporal_gt:batch_video_gt,
                            vname:batch_video_name,
                            label:batch_video_label,
                            duration:batch_video_duration})
                #------ Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, out_g_loss2 = sess.run([g_optim, loss_adv_gen],
                  feed_dict={feature_seg:batch_anchor_feature_mome,
                            temporal_gt_seg:batch_video_gt_mome,
                            vname_seg:batch_video_name_mome,
                            label_seg:batch_video_label_mome,
                            duration_seg:batch_video_duration_mome,
                            ratio_seg: batch_video_ratio_mome,
                            position_seg:batch_video_position_mome,
                            feature:batch_anchor_feature,
                            temporal_gt:batch_video_gt,
                            vname:batch_video_name,
                            label:batch_video_label,
                            duration:batch_video_duration})         
                
                # 6. train localization on synthetic moments (Kinetics) to optimize 1D backbone
                _,out_loss_bone=sess.run([optimizer_bone,loss_bone], 
                                 feed_dict={feature_seg:batch_anchor_feature_mome,
                                            temporal_gt_seg:batch_video_gt_mome,
                                            vname_seg:batch_video_name_mome,
                                            label_seg:batch_video_label_mome,
                                            duration_seg:batch_video_duration_mome,
                                            ratio_seg: batch_video_ratio_mome,
                                            position_seg:batch_video_position_mome,
                                            LR:lr_now})  
                # ----------------------------------------------------------------------------------------------------------------------#



                for key in info_keys:
                    mini_info[key].append(out_loss_untrim[key])
                    mini_info_bone[key].append(out_loss_bone[key])
                for key in info_adv_dis_keys:
                    mini_adv_dis_info[key].append(out_d_loss[key])
                for key in info_adv_gen_keys:
                    mini_adv_gen_info[key].append(out_g_loss1[key])
                    mini_adv_gen_info[key].append(out_g_loss2[key])
                if idx % 100 == 0: 
                    print("Batch-%d LR: %.08f ProLoss: %.04f Loc Loss: %.04f Prop Loss: %.04f IoU Loss: %.4f Adv-Dis: %.04f Adv-G: %.04f Total loss: %.04f Cls Loss: %.04f Cls Acc: %0.4f"%
                    (idx,lr_now,out_loss_untrim["Prop_loss"],out_loss_untrim["loss_loc"],out_loss_untrim["loss_prop"],out_loss_untrim["loss_iou"],
                    out_d_loss["loss_dis"],(out_g_loss1["loss_gen"]+out_g_loss2["loss_gen"])/2,
                    out_loss_untrim["Total_loss"],out_loss_untrim["loss_cls"],out_loss_untrim["acc_cls"]))
                    print("Batch-%d LR: %.08f BProLoss: %.04f BLoc Loss: %.04f BProp Loss: %.04f BIoU Loss: %.4f BCls Loss: %.04f BCls Acc: %0.4f"%
                    (idx,lr_now,out_loss_bone["Prop_loss"],out_loss_bone["loss_loc"],out_loss_bone["loss_prop"],out_loss_bone["loss_iou"],out_loss_bone["loss_cls"],out_loss_bone["acc_cls"]))
                    fw_log.write("Batch-%d LR: %.08f ProLoss: %.04f Loc Loss: %.04f Prop Loss: %.04f IoU Loss: %.4f Adv-Dis: %.04f Adv-G: %.04f Total loss: %.04f Cls Loss: %.04f Cls Acc: %0.4f\n"%
                    (idx,lr_now,out_loss_untrim["Prop_loss"],out_loss_untrim["loss_loc"],out_loss_untrim["loss_prop"],out_loss_untrim["loss_iou"],
                    out_d_loss["loss_dis"],(out_g_loss1["loss_gen"]+out_g_loss2["loss_gen"])/2,
                    out_loss_untrim["Total_loss"],out_loss_untrim["loss_cls"],out_loss_untrim["acc_cls"]))
                    fw_log.write("Batch-%d LR: %.08f BProLoss: %.04f BLoc Loss: %.04f BProp Loss: %.04f BIoU Loss: %.4f BCls Loss: %.04f BCls Acc: %0.4f\n"%
                    (idx,lr_now,out_loss_bone["Prop_loss"],out_loss_bone["loss_loc"],out_loss_bone["loss_prop"],out_loss_bone["loss_iou"],out_loss_bone["loss_cls"],out_loss_bone["acc_cls"]))

                global_iternum += config.batch_size
                total_batch_idx += 1

                if total_batch_idx % snapshot_step == 0:
    
                    for key in info_keys:
                        train_info[key].append(np.mean(mini_info[key]))
                        train_info_bone[key].append(np.mean(mini_info_bone[key]))
                    for key in info_adv_dis_keys:
                        train_adv_dis_info[key].append(np.mean(mini_adv_dis_info[key]))        
                    for key in info_adv_gen_keys:
                        train_adv_gen_info[key].append(np.mean(mini_adv_gen_info[key])) 

                    print('*********************************************************')
                    print("STEP-%d Train ProLoss: %.04f Loc Loss: %.04f Prop Loss: %.04f IoU Loss: %.04f Adv-Dis: %.04f Adv-G: %.04f Train Total loss: %.04f Cls Loss: %.04f Cls Acc: %.04f" 
                          %(total_batch_idx,train_info["Prop_loss"][-1],train_info["loss_loc"][-1],train_info["loss_prop"][-1],train_info["loss_iou"][-1],
                          train_adv_dis_info["loss_dis"][-1],train_adv_gen_info["loss_gen"][-1],
                          train_info["Total_loss"][-1],train_info["loss_cls"][-1],train_info["acc_cls"][-1]))
                    print("STEP-%d Train BProLoss: %.04f BLoc Loss: %.04f BProp Loss: %.04f BIoU Loss: %.04f Train BTotal loss: %.04f BCls Loss: %.04f BCls Acc: %.04f" 
                          %(total_batch_idx,train_info_bone["Prop_loss"][-1],train_info_bone["loss_loc"][-1],train_info_bone["loss_prop"][-1],train_info_bone["loss_iou"][-1],
                          train_info_bone["Total_loss"][-1],train_info_bone["loss_cls"][-1],train_info_bone["acc_cls"][-1]))
                    print('*********************************************************')
                    fw_log.write('*********************************************************\n')
                    fw_log.write("STEP-%d Train ProLoss: %.04f Loc Loss: %.04f Prop Loss: %.04f IoU Loss: %.04f Adv-Dis: %.04f Adv-G: %.04f Train Total loss: %.04f Cls Loss: %.04f Cls Acc: %.04f\n" 
                          %(total_batch_idx,train_info["Prop_loss"][-1],train_info["loss_loc"][-1],train_info["loss_prop"][-1],train_info["loss_iou"][-1],
                          train_adv_dis_info["loss_dis"][-1],train_adv_gen_info["loss_gen"][-1],
                          train_info["Total_loss"][-1],train_info["loss_cls"][-1],train_info["acc_cls"][-1]))
                    fw_log.write("STEP-%d Train BProLoss: %.04f BLoc Loss: %.04f BProp Loss: %.04f BIoU Loss: %.04f Train BTotal loss: %.04f BCls Loss: %.04f BCls Acc: %.04f\n" 
                          %(total_batch_idx,train_info_bone["Prop_loss"][-1],train_info_bone["loss_loc"][-1],train_info_bone["loss_prop"][-1],train_info_bone["loss_iou"][-1],
                          train_info_bone["Total_loss"][-1],train_info_bone["loss_cls"][-1],train_info_bone["acc_cls"][-1]))
                    fw_log.write('*********************************************************\n')
                    """ save model """ 
                    model_saver.save(sess,"%s/aher_adv_model_checkpoint-step_%d"%(model_output_dir,total_batch_idx)) 

    fw_log.close()
    data_generate.stop_process()
    data_moment_generate.stop_process()

    
