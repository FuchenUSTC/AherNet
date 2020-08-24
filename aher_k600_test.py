# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import aher_anet
import tf_utils
from datetime import datetime
import time
from data_loader import *
import tf_extended as tfe
import os
import sys
import pandas as pd
from multiprocessing import Process,Queue
import multiprocessing
import math
import random
from evaluation import get_proposal_performance

FLAGS = tf.app.flags.FLAGS
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

tf.app.flags.DEFINE_float('loss_alpha', 1., 'Alpha parameter in the loss function.') 
tf.app.flags.DEFINE_float('match_threshold', 0.65, 'Matching threshold in the loss function.') # 0.65
tf.app.flags.DEFINE_float('neg_match_threshold', 0.35, 'Negative threshold in the loss function.') # 0.3
tf.app.flags.DEFINE_float('negative_ratio', 1., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float('label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_integer('output_idx', 1, 'The output index.')
tf.app.flags.DEFINE_integer('training_epochs',7,'The training epochs number')
tf.app.flags.DEFINE_float('cls_weights', 2.0, 'The weight for the classification.') # 0.1
tf.app.flags.DEFINE_float('iou_weights', 25.0, 'The weight for the iou prediction.')  # 5.0
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001,'Initial learning rate.')
# Flags for box generation
tf.app.flags.DEFINE_float('select_threshold', 0.0, 'Selection threshold.')
tf.app.flags.DEFINE_float('nms_threshold', 0.90, 'Non-Maximum Selection threshold.')
tf.app.flags.DEFINE_integer('select_top_k', 765, 'Select top-k detected bounding boxes.')
tf.app.flags.DEFINE_integer('keep_top_k', 100, 'Keep top-k detected objects.')
tf.app.flags.DEFINE_bool('cls_flag', False, 'Utilize classification score.')

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

def AHER_Detection_Inference(aher_anet_model,aher_anet_anchor,
                   feature,vname,label,duration, clsweights, clsbias, reuse, n_class,cls_suffix='_anet'):
    """ Inference bbox of sigle shot action localization
        feature:      batch_size x 512 x 4069
        vname:        batch_size x 1
        label:        batch_size x 1
        duration:     batch_size x 1
    """   

    predictions, localisation, logits, proplogits, proppredictions, iouprediction, end_points \
       = aher_anet_model.net_prop_iou(feature,clsweights,clsbias,is_training=True,reuse=reuse,num_classes=n_class,cls_suffix=cls_suffix)

    # decode bounding box and get scores
    localisation = aher_anet_model.bboxes_decode_logits(localisation, duration ,aher_anet_anchor, predictions)
    if FLAGS.cls_flag:
        rscores, rbboxes = aher_anet_model.detected_bboxes_classwise(
                                    proppredictions, localisation,                                        
                                    select_threshold=FLAGS.select_threshold,
                                    nms_threshold=FLAGS.nms_threshold,
                                    clipping_bbox=None,
                                    top_k=FLAGS.select_top_k,
                                    keep_top_k=FLAGS.keep_top_k,
                                    iou_flag=False)
    else:
        rscores, rbboxes = aher_anet_model.detected_bboxes_classwise(
                                    iouprediction, localisation,                                        
                                    select_threshold=FLAGS.select_threshold,
                                    nms_threshold=FLAGS.nms_threshold,
                                    clipping_bbox=None,
                                    top_k=FLAGS.select_top_k,
                                    keep_top_k=FLAGS.keep_top_k,
                                    iou_flag=True)
    # compute pooling score
    lshape = tfe.get_shape(predictions[0], 8)
    num_classes = lshape[-1]
    batch_size = lshape[0]
    fprediction = []
    for i in range(len(predictions)):
        fprediction.append(tf.reshape(predictions[i], [-1, num_classes]))
    predictions = tf.concat(fprediction, axis=0)
    avergeprediction = tf.reduce_mean(predictions,axis=0)
    labelid = tf.argmax(avergeprediction, 0)
    argmaxid = tf.argmax(predictions, 1)

    prebbox={"rscores":rscores,"rbboxes":rbboxes,"label":labelid,"avescore":avergeprediction, \
             "rawscore":predictions,"argmaxid":argmaxid}
    return prebbox

class Config(object):
    def __init__(self):
        self.learning_rates=[0.001]*100+[0.0001]*100
        #self.training_epochs = 15
        self.training_epochs = 12
        self.n_inputs = 2048
        self.batch_size = 8
        self.input_steps=512
        self.input_resize_steps=512
        self.gt_hold_num = 25
        self.batch_size_val=1

if __name__ == "__main__":
    """ define the input and the network""" 
    config = Config()
    LR= tf.placeholder(tf.float32)

    #--------------------------------------------Restore Folder and Feature Folder----------------------------------------------#
    eva_step_id = [8000,9000,10000] # The model id array to be evaluated
    output_folder_dir = 'p3d_k600_models/AHER_k600_gen_adv' # The restore folder to output localization results
    csv_dir = '/data/Kinetics-600/csv_p3d_clip_val_untrim_512' # validation clip feature for Kinetics-600
    csv_oris_dir = ''
    #------------------------------------------------------------------------------------------------------------------------------#




    #-------------------------------------------------------Test Placeholder-----------------------------------------------------#
    # train placehold
    feature_seg = tf.placeholder(tf.float32, shape=(config.batch_size,config.input_steps,config.n_inputs))
    temporal_gt_seg = tf.placeholder(tf.float32, shape=(config.batch_size, config.gt_hold_num, 2))
    vname_seg = tf.placeholder(tf.string, shape=(config.batch_size))
    label_seg = tf.placeholder(tf.int32, shape=(config.batch_size))
    duration_seg = tf.placeholder(tf.float32,shape=(config.batch_size)) 

    # val placehold 
    feature_val = tf.placeholder(tf.float32, shape=(config.batch_size_val,config.input_steps,config.n_inputs))
    temporal_gt_val = tf.placeholder(tf.float32, shape=(config.batch_size_val, config.gt_hold_num, 2))
    vname_val = tf.placeholder(tf.string, shape=(config.batch_size_val))
    label_val = tf.placeholder(tf.int32, shape=(config.batch_size_val))
    duration_val = tf.placeholder(tf.float32,shape=(config.batch_size_val))    
    #------------------------------------------------------------------------------------------------------------------------------#    

    


    #----------------------------------------------- AherNet Structure ------------------------------------------------------------#
    aher_anet_model,aher_anet_anchor = AHER_init()
    # Initialize the backbone
    predictions_gene_seg, localisation_gene_seg, logits_gene_seg, proplogits_gene_seg, iouprediction_gene_seg, \
    clsweights_ki, clsbias_ki, gscore_gene_seg, \
    giou_gene_seg, gclasses_gene_seg, glocalisations_gene_seg, \
    end_points_gene_seg \
    = AHER_Predictor_Cls(aher_anet_model,aher_anet_anchor,feature_seg, \
    temporal_gt_seg,vname_seg,label_seg,duration_seg,reuse=False,class_num=600,cls_suffix='_anchor')

    # localization inference
    bbox = AHER_Detection_Inference(aher_anet_model,aher_anet_anchor,feature_val, \
    vname_val,label_val,duration_val, clsweights_ki, clsbias_ki, reuse=True, n_class=600,cls_suffix='_anchor')    
    #------------------------------------------------------------------------------------------------------------------------------#




    AHER_trainable_variables=tf.trainable_variables() 

    """ Init tf""" 
    model_saver=tf.train.Saver(var_list=AHER_trainable_variables,max_to_keep=80)
    full_vars = []
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement =True
    sess=tf.InteractiveSession(config=tf_config)



    #---------------------------------------------------- Test Data Input List -------------------------------------------------------#
    data_generate_val = KineticsDatasetLoadTrimList('val_moment',csv_dir,csv_oris_dir,25,512,df_file='./cvs_records_k600/k600_train_val_info.csv')
    print('Load train feature, wait...')
    while True:
        if len(data_generate_val.train_feature_list) == 6459:
            data_generate_val.stop_process()
            time.sleep(30)
            break
    print('Wait done.')
    print('The val feature len is %d'%(len(data_generate_val.train_feature_list)))  
    dataset_val = tf.data.Dataset.from_generator(data_generate_val.gen,
    (tf.float32, tf.float32, tf.string, tf.int32, tf.float32),
    (tf.TensorShape([512, 2048]),tf.TensorShape([25, 2]),tf.TensorShape([]),tf.TensorShape([]),tf.TensorShape([])))
    dataset_val = dataset_val.batch(config.batch_size_val)
    batch_num_val = int(len(data_generate_val.train_feature_list) / config.batch_size_val)    
    iterator_val = dataset_val.make_one_shot_iterator()
    feature_g_val, \
    video_gt_g_val, \
    video_name_g_val, \
    video_label_g_val, \
    video_duration_g_val  = iterator_val.get_next() 
    #---------------------------------------------------------------------------------------------------------------------------------#


    
    #---------------------------------------------------- Category Information --------------------------------------------------------#
    # load category information
    cateidx_name = {}
    category_info = pd.read_csv('k600_category.txt',sep='\n',header=None)
    for i in range(len(category_info)):
        name = category_info.loc[i][0]
        cateidx_name[i] = name 
     #---------------------------------------------------------------------------------------------------------------------------------------#




    #---------------------------------------------------- Detection Result Extraction --------------------------------------------------------#
    fw_log = open('%s/log_res_k600_gene_dec.txt'%(output_folder_dir),'w',1)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        for epoch in eva_step_id:
            print('Restore model %s/aher_adv_model_checkpoint-step_%d'%(output_folder_dir,epoch))
            model_saver.restore(sess,"%s/aher_adv_model_checkpoint-step_%d"%(output_folder_dir,epoch))
            """ Validation""" 
            hit_video_num = 0
            total_video_num = 0   
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + \
            "----Epoch-%d Val set validata start." %(epoch))
            fw_proposal = open('%s/results_aher_k600_gendec_step-%d.json'%(output_folder_dir,epoch),'w')
            fw_proposal.write('{\"version\": \"VERSION 1.3\", \"results\": {')  
            for idx in range(batch_num_val):
                feature_batch_val, \
                video_gt_batch_val, \
                video_name_batch_val, \
                video_label_batch_val, \
                video_duration_batch_val = sess.run([feature_g_val,video_gt_g_val,video_name_g_val,video_label_g_val,video_duration_g_val])
                out_bbox=sess.run(bbox,feed_dict={feature_val:feature_batch_val,
                                                  vname_val:video_name_batch_val,
                                                  label_val:video_label_batch_val,
                                                  duration_val:video_duration_batch_val})
                predict_score = out_bbox['rscores']
                predict_bbox  = out_bbox['rbboxes']
                average_cls_score = out_bbox['avescore']
                cls_id = out_bbox['label']
                raw_score = out_bbox['rawscore']
                argmaxid = out_bbox['argmaxid']
                real_label = video_label_batch_val[0]
                if real_label == cls_id: hit_video_num += 1
                total_video_num += 1
                cateids = 1
                write_first = 0
                # write the json file
                if idx!=0: fw_proposal.write(', ')
                fw_proposal.write('\"%s\":['%(video_name_batch_val[0].decode("utf-8")))
                for kk in range(len(predict_score[1][0])):
                    # compute the ranking score for each localization result
                    score  = predict_score[cateids][0][kk] * average_cls_score[cls_id]
                    start_time = predict_bbox[cateids][0][kk][0]
                    end_time = predict_bbox[cateids][0][kk][1]
                    if score > 0 and end_time - start_time > 0:
                        if write_first == 0:
                            fw_proposal.write('{\"score\": %f, \"segment\": [%f, %f], \"label\": \"%s\"}'% \
                            (score,start_time,end_time,cateidx_name[cls_id])) # 
                        else:
                            fw_proposal.write(', {\"score\": %f, \"segment\": [%f, %f], \"label\": \"%s\"}'% \
                            (score,start_time,end_time,cateidx_name[cls_id]))
                        write_first += 1     
                fw_proposal.write(']')
            fw_proposal.write('}, \"external_data\": {}}')
            fw_proposal.close()   
            # evaluation for temporal action proposal
            an_ar,recall = get_proposal_performance.evaluate_return_area('data_split/k600_val_action.json', \
                   '%s/results_aher_k600_gendec_step-%d.json'%(output_folder_dir,epoch))
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + \
            "----Epoch-%d Val set validata finished." %(epoch))
            accuracy_video = hit_video_num / total_video_num
        
            print('*********************************************************')
            print("Epoch-%d Val AUC: %.04f AverageRecall: %04f Accuracy: %04f" %(epoch,an_ar,recall,accuracy_video))
            print('*********************************************************')
            fw_log.write('*********************************************************\n')
            fw_log.write("Epoch-%d Val AUC: %.04f AverageRecall: %04f Accuracy: %04f\n" %(epoch,an_ar,recall,accuracy_video))
            fw_log.write('*********************************************************\n')        
    
    fw_log.close()
        