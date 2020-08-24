import random
import numpy as np
import pandas as pd
import json
import scipy
import time
import tensorflow as tf
from multiprocessing import Process,Queue,JoinableQueue
import multiprocessing

x_tdim = 512
fea_dim_th = 2048

def pool_fea_interpolate_th(fea_temp,feature_count,duration,fnumber,resize_temporal_dim=100,pool_type="mean"):

    num_bin = 1
    num_sample_bin=3
    
    num_prop = resize_temporal_dim
    video_frame = fnumber
    video_second = duration

    data = fea_temp[0:feature_count]
    

    feature_frame=feature_count*8
    corrected_second=float(feature_frame)/video_frame*video_second
    fps=float(video_frame)/video_second
    st=8/fps

    if feature_count==1:
        video_feature=np.stack([data]*num_prop)
        video_feature=np.reshape(video_feature,[num_prop,fea_dim_th])
        return video_feature

    x=[st/2+ii*st for ii in range(feature_count)]
    f=scipy.interpolate.interp1d(x,data,axis=0)
        
    video_feature=[]
    zero_sample=np.zeros(num_bin*fea_dim_th)
    tmp_anchor_xmin=[1.0/num_prop*i for i in range(num_prop)]
    tmp_anchor_xmax=[1.0/num_prop*i for i in range(1,num_prop+1)]        
    
    num_sample=num_bin*num_sample_bin
    for idx in range(num_prop):
        xmin=max(x[0]+0.0001,tmp_anchor_xmin[idx]*corrected_second)
        xmax=min(x[-1]-0.0001,tmp_anchor_xmax[idx]*corrected_second)
        if xmax<x[0]:
            video_feature.append(zero_sample)
            continue
        if xmin>x[-1]:
            video_feature.append(zero_sample)
            continue
            
        plen=(xmax-xmin)/(num_sample-1)
        x_new=[xmin+plen*ii for ii in range(num_sample)]
        y_new=f(x_new)
        y_new_pool=[]
        for b in range(num_bin):
            tmp_y_new=y_new[num_sample_bin*b:num_sample_bin*(b+1)]
            if pool_type=="mean":
                tmp_y_new=np.mean(y_new,axis=0)
            elif pool_type=="max":
                tmp_y_new=np.max(y_new,axis=0)
            y_new_pool.append(tmp_y_new)
        y_new_pool=np.stack(y_new_pool)
        y_new_pool=np.reshape(y_new_pool,[-1])
        video_feature.append(y_new_pool)
    video_feature=np.stack(video_feature)
    return video_feature

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def getDatasetDict(df_file='./cvs_records/anet_train_val_info.csv'):
    """Load dataset file
    """
    df=pd.read_csv(df_file)
    json_data= load_json("./cvs_records_anet/anet_annotation_json.json")
    database=json_data
    train_dict={}
    val_dict={}
    test_dict={}
    miss_count = 0
    for i in range(len(df)):
        video_name=df.video.values[i]
        if video_name not in database.keys():
            miss_count += 1
            continue
        video_info=database[video_name]
        video_new_info={}
        video_new_info['duration_frame']=video_info['duration_frame']
        video_new_info['duration_second']=video_info['duration_second']
        video_new_info["feature_frame"]=video_info['feature_frame']
        video_subset=df.subset.values[i]
        video_new_info['annotations']=video_info['annotations']
        if video_subset=="training":
            train_dict[video_name]=video_new_info
        elif video_subset=="validation":
            val_dict[video_name]=video_new_info
        elif video_subset=="testing":
            test_dict[video_name]=video_new_info
    print('Miss video annotation: %d'%(miss_count))
    return train_dict,val_dict,test_dict

def getDatasetDictK600(df_file='./cvs_records_k600/k600_train_val_info.csv'):
    """Load dataset file
    """
    df=pd.read_csv(df_file)
    json_data= load_json("./cvs_records_k600/k600_annotation_json.json")
    database=json_data
    train_dict={}
    val_dict={}
    test_dict={}
    miss_count = 0
    for i in range(len(df)):
        video_name=df.video.values[i]
        if video_name not in database.keys():
            miss_count += 1
            continue
        video_info=database[video_name]
        video_new_info={}
        video_new_info['duration_frame']=video_info['duration_frame']
        video_new_info['duration_second']=video_info['duration_second']
        video_new_info["feature_frame"]=video_info['feature_frame']
        video_subset=df.subset.values[i]
        video_new_info['annotations']=video_info['annotations']
        if video_subset=="training":
            train_dict[video_name]=video_new_info
        elif video_subset=="validation":
            val_dict[video_name]=video_new_info
        elif video_subset=="testing":
            test_dict[video_name]=video_new_info
    print('Miss video annotation: %d'%(miss_count))
    return train_dict,val_dict,test_dict

def getSplitsetDict(df_file='./cvs_records/anet_train_val_info.csv'):
    """Load dataset file
       ActivityNet: 
       untrim set class number: 87
       moment set class number: 113
    """
    cate_info = pd.read_csv('activitynet_train_val_gt.txt',sep = ' ',header = None)
    cate_dict = {}
    for v_id in range(len(cate_info)):
        v = cate_info.loc[v_id][0]
        cate = cate_info.loc[v_id][2]
        cate_dict[v] = cate
    
    moment_info = pd.read_csv('data_split/label_map_moment.txt',sep = '\n',header = None)
    moment_cate = []
    for idm in range(len(moment_info)):
        moment_cate.append(moment_info.loc[idm][0])

    untrim_info = pd.read_csv('data_split/label_map_untrim.txt',sep = '\n',header = None)
    untrim_cate = []
    for idm in range(len(untrim_info)):
        untrim_cate.append(untrim_info.loc[idm][0])

    df=pd.read_csv("./cvs_records/anet_train_val_info.csv")
    json_data= load_json("./cvs_records/anet_annotation_json.json")
    database=json_data
    train_untrim_dict={}
    train_moment_dict={}
    val_untrim_dict={}
    val_moment_dict={}
    miss_count = 0
    for i in range(len(df)):
        video_name=df.video.values[i]
        if video_name not in database.keys():
            miss_count += 1
            continue
        video_info=database[video_name]
        video_cate=cate_dict[video_name]
        video_new_info={}
        video_new_info['duration_frame']=video_info['duration_frame']
        video_new_info['duration_second']=video_info['duration_second']
        video_new_info["feature_frame"]=video_info['feature_frame']
        video_subset=df.subset.values[i]
        video_new_info['annotations']=video_info['annotations']

        if video_subset=="training" and video_cate in untrim_cate:
            train_untrim_dict[video_name]=video_new_info
        if video_subset=="training" and video_cate in moment_cate:
            train_moment_dict[video_name]=video_new_info
        if video_subset=="validation" and video_cate in untrim_cate:
            val_untrim_dict[video_name]=video_new_info
        if video_subset=="validation" and video_cate in moment_cate:
            val_moment_dict[video_name]=video_new_info
    
    print('Miss video annotation: %d'%(miss_count))
    return train_untrim_dict,train_moment_dict,val_untrim_dict,val_moment_dict

class AnetDatasetLoadForeQueue():
    def __init__(self,dataSet,csv_dir,csv_oris_dir,gt_hold_num,x_tdim,df_file='./cvs_records_anet/anet_train_val_reduce_info.csv',untrim_start=87,resize_dim=512):   
        self.train_dict,self.val_dict,self.test_dict=getDatasetDict(df_file=df_file)
        if dataSet == 'train':
            self.train_list = list(self.train_dict.keys())
        else:
            self.train_list = list(self.val_dict.keys())
        self.train_list_copy = self.train_list
        cate_info = pd.read_csv('activitynet_train_val_gt.txt',sep = ' ',header = None)
        self.cate_dict = {}
        for v_id in range(len(cate_info)):
            v = cate_info.loc[v_id][0]
            cate = cate_info.loc[v_id][2]
            self.cate_dict[v] = cate
        self.shuffle()
        self.resize_dim = resize_dim
        self.dataSet = dataSet
        self.csv_dir = csv_dir
        self.csv_oris_dir = csv_oris_dir
        self.gt_hold_num = gt_hold_num
        self.x_tdim = x_tdim
        self.num_samples = len(self.train_list)
        self.batch_size = 1
        self.queue = Queue(maxsize=1536) #Joinable
        self.train_feature_list = multiprocessing.Manager().list()
        self.process_num = 32
        self.process_array = []
        self.start_train_idx = 0
        for pid in range(self.process_num):
            pid_num =  int(self.num_samples / self.process_num)
            start_idx = pid*pid_num
            end_idx = (pid+1)*pid_num
            if pid == (self.process_num - 1):
                if end_idx < self.num_samples: end_idx = self.num_samples
            t = Process(target=self.load_queue,args=(start_idx,end_idx))
            self.process_array.append(t)
            t.start()
    def shuffle(self):
        randperm = np.random.permutation(len(self.train_list))
        self.train_list = [self.train_list_copy[int(randperm[idx])] for idx in range(len(randperm))]
    def load_single_feature(self,start):
        result = []
        name = self.train_list[start]
        s_bbox = []
        r_bbox = []
        video_info=self.train_dict[name]
        video_frame=video_info['duration_frame']
        video_second=video_info['duration_second']
        feature_frame=video_info['feature_frame']
        feature_num = int(feature_frame / 8)
        corrected_second=float(feature_frame)/video_frame*video_second
        video_labels=video_info['annotations'] 
        vlabels = self.cate_dict[name]
        for j in range(len(video_labels)):
            tmp_info=video_labels[j]
            tmp_start=tmp_info['segment'][0]
            tmp_end=tmp_info['segment'][1]
            tmp_start=max(min(1,tmp_start/corrected_second),0)
            tmp_end=max(min(1,tmp_end/corrected_second),0)
            tmp_fea_start = tmp_start*self.x_tdim
            tmp_fea_end = tmp_end*self.x_tdim
            s_bbox.append([tmp_fea_start,tmp_fea_end])
            tmp_len = tmp_end - tmp_start
            if tmp_len*feature_num >= 32: r_bbox.append([tmp_start,tmp_end])

            tmp_df=pd.read_csv(self.csv_oris_dir+"/"+name+".csv",skiprows=1,header=None,sep=',')
            tmp_values = tmp_df.values[:,:]
            for m in range(len(r_bbox)):
                tmp_start_fea,tmp_end_fea = int(r_bbox[m][0]*feature_num),int(r_bbox[m][1]*feature_num)
                proposal_fea = tmp_values[tmp_start_fea:tmp_end_fea,:]
                proposal_duration,proposal_frame = (r_bbox[m][1]-r_bbox[m][0])*video_second,(r_bbox[m][1]-r_bbox[m][0])*video_frame
                proposal_inter_fea = pool_fea_interpolate_th(proposal_fea,len(proposal_fea),proposal_duration,proposal_frame,self.resize_dim)
                result.append([proposal_inter_fea,[[128.0,384.0]],name,vlabels,proposal_duration*2])        
            return result
    def load_queue(self,start,end):
        bindex = 0
        lendata = end - start
        while True:
            t = self.load_single_feature(bindex+start)
            if self.dataSet == 'train_untrim' or \
            self.dataSet == 'train_momwhole' or \
            self.dataSet == 'val_untrim' or \
            self.dataSet == 'val_moment': 
                self.queue.put(t)
            else:
                for z in range(len(t)):
                    self.queue.put(t[z])
            bindex += 1
            if bindex == lendata:
                bindex = 0
    def gen(self):
        while True:
            t = self.queue.get()
            yield t[0],t[1],t[2],t[3],t[4]    
    def stop_process(self):
        for t in self.process_array:
            t.terminate()
            t.join()
        print('Stop the process sucess.')

class KineticsDatasetLoadTrimMultiProcess():
    def __init__(self,dataSet,csv_dir,csv_oris_dir,gt_hold_num,x_tdim,df_file='./cvs_records_k600/k600_train_val_info.csv',resize_dim=256,ratio_sid=1,ratio_eid=9):
        train_video_dict, val_video_dict, test_video_dict=getDatasetDictK600(df_file=df_file)
        if "train" in dataSet:
            self.train_dict=train_video_dict
        else:
            self.train_dict=val_video_dict
        self.dataSet = dataSet
        self.resize_dim = resize_dim
        self.ratio_sid = ratio_sid
        self.ratio_eid = ratio_eid
        self.train_list = list(self.train_dict.keys())
        self.train_list_copy = self.train_list 
        print('The %s video num is: %d'%(self.dataSet,len(self.train_list)))
        print('Load k600 train val gt info...')
        cate_info = pd.read_csv('k600_train_val_gt.txt',sep = ' ',header = None)
        self.cate_dict = {}
        for v_id in range(len(cate_info)):
            v = cate_info.loc[v_id][0]
            cate = cate_info.loc[v_id][2]
            self.cate_dict[v] = cate  
        print('Load Done.')
        self.shuffle()
        self.csv_dir = csv_dir
        self.csv_oris_dir = csv_oris_dir
        self.gt_hold_num = gt_hold_num
        self.x_tdim = x_tdim
        self.num_samples = len(self.train_list)
        self.batch_size = 1
        #self.train_feature_list = multiprocessing.Manager().list() 
        self.process_num = 32
        self.process_array = []
        self.queue = Queue(maxsize=1536) #Joinable
        self.start_train_idx = 0
        for pid in range(self.process_num):
            pid_num =  int(self.num_samples / self.process_num)
            start_idx = pid*pid_num
            end_idx = (pid+1)*pid_num
            if pid == (self.process_num - 1):
                if end_idx < self.num_samples: end_idx = self.num_samples
            t = Process(target=self.load_queue,args=(start_idx,end_idx))
            self.process_array.append(t)
            t.start()
    def shuffle(self):
        randperm = np.random.permutation(len(self.train_list))
        self.train_list = [self.train_list_copy[int(randperm[idx])] for idx in range(len(randperm))]
    def load_single_feature(self,start):
        result = []
        name = self.train_list[start]
        s_bbox = []
        r_bbox = []
        label_bbox=[]
        slabel_box = []
        rlabel_bbox = []
        video_info=self.train_dict[name]
        video_frame=video_info['duration_frame']
        video_second=video_info['duration_second']
        feature_frame=video_info['feature_frame']
        feature_num = int(feature_frame / 8)
        corrected_second=float(feature_frame)/video_frame*video_second
        video_labels=video_info['annotations'] 
        vlabels = self.cate_dict[name]
        for j in range(len(video_labels)):
            tmp_info=video_labels[j]
            tmp_start=tmp_info['segment'][0]
            tmp_end=tmp_info['segment'][1]
            tmp_label=tmp_info['label']
            tmp_start=max(min(1,tmp_start/corrected_second),0)
            tmp_end=max(min(1,tmp_end/corrected_second),0)
            tmp_fea_start = tmp_start*self.x_tdim
            tmp_fea_end = tmp_end*self.x_tdim
            s_bbox.append([tmp_fea_start,tmp_fea_end])
            tmp_len = tmp_end - tmp_start
            if tmp_len*feature_num >= 8: 
                r_bbox.append([tmp_start,tmp_end])
        if  self.dataSet == 'val_moment': 
            tmp_df=pd.read_csv(self.csv_dir+"/"+name+".csv",skiprows=1,header=None,sep=',')
            anchor_feature = tmp_df.values[:,:]  
            for kk in range(len(s_bbox),self.gt_hold_num):
                s_bbox.append([-1.0,-1.0])         
            result.append(anchor_feature)
            result.append(s_bbox)
            result.append(name)
            result.append(vlabels)
            result.append(video_second)
            return result
        else:
            tmp_df=pd.read_csv(self.csv_oris_dir+"/"+name+".csv",skiprows=1,header=None,sep=',')
            tmp_values = tmp_df.values[:,:]
            for m in range(len(r_bbox)):
                tmp_start_fea,tmp_end_fea = int(r_bbox[m][0]*feature_num),int(r_bbox[m][1]*feature_num)
                proposal_fea = tmp_values[tmp_start_fea:tmp_end_fea,:]
                proposal_duration,proposal_frame = (r_bbox[m][1]-r_bbox[m][0])*video_second,(r_bbox[m][1]-r_bbox[m][0])*video_frame
                proposal_inter_fea = pool_fea_interpolate_th(proposal_fea,len(proposal_fea),proposal_duration,proposal_frame,self.resize_dim)
                result.append([proposal_inter_fea,[[128.0,384.0]],name,vlabels,proposal_duration*2])        
            return result
    def load_queue(self,start,end):
        bindex = 0
        lendata = end - start
        while True:
            t = self.load_single_feature(bindex+start)
            if self.dataSet == 'val_moment':
                self.queue.put(t)
            else:
                for z in range(len(t)):
                    self.queue.put(t[z])
            bindex += 1
            if bindex == lendata: bindex = 0
    def gen(self):
        while True:
            if self.queue.empty():
                print('The Kinetics %s Queue is empty, loading feature...'%(self.dataSet))
                time.sleep(10)
            else:
                t = self.queue.get_nowait()
                yield t[0],t[1],t[2],t[3],t[4] 
    def stop_process(self):
        for t in self.process_array:
            t.terminate()
            t.join()
        print('Stop the process sucess.')
    def gen_pos_random(self):
        while True:
            t = self.queue.get()
            yield t[0],t[1],t[2],t[3],t[4],random.randint(self.ratio_sid,self.ratio_eid),random.randint(1,19)

class AnetDatasetLoadMultiProcessQueue():
    def __init__(self,dataSet,csv_dir,gt_hold_num,x_tdim,df_file='./cvs_records_anet/anet_train_val_reduce_info.csv'):
        self.train_dict,self.val_dict,self.test_dict=getDatasetDict(df_file=df_file)
        if dataSet == 'train':
            self.train_list = list(self.train_dict.keys())
        else:
            self.train_list = list(self.val_dict.keys())
        self.train_list_copy = self.train_list
        cate_info = pd.read_csv('activitynet_train_val_gt.txt',sep = ' ',header = None)
        self.cate_dict = {}
        for v_id in range(len(cate_info)):
            v = cate_info.loc[v_id][0]
            cate = cate_info.loc[v_id][2]
            self.cate_dict[v] = cate
        self.shuffle()
        self.dataSet = dataSet
        self.csv_dir = csv_dir
        self.gt_hold_num = gt_hold_num
        self.x_tdim = x_tdim
        self.num_samples = len(self.train_list)
        self.batch_size = 1
        self.queue = Queue(maxsize=1536) #Joinable
        self.train_feature_list = multiprocessing.Manager().list()
        self.process_num = 32
        self.process_array = []
        self.start_train_idx = 0
        for pid in range(self.process_num):
            pid_num =  int(self.num_samples / self.process_num)
            start_idx = pid*pid_num
            end_idx = (pid+1)*pid_num
            if pid == (self.process_num - 1):
                if end_idx < self.num_samples: end_idx = self.num_samples
            t = Process(target=self.load_queue,args=(start_idx,end_idx))
            self.process_array.append(t)
            t.start()
    def shuffle(self):
        randperm = np.random.permutation(len(self.train_list))
        self.train_list = [self.train_list_copy[int(randperm[idx])] for idx in range(len(randperm))]
    def load_single_feature(self,start):
        result = []
        name = self.train_list[start]
        s_bbox = []
        if self.dataSet == 'train':
            video_info=self.train_dict[name]
        else:
            video_info=self.val_dict[name]
        video_frame=video_info['duration_frame']
        video_second=video_info['duration_second']
        feature_frame=video_info['feature_frame']
        corrected_second=float(feature_frame)/video_frame*video_second
        video_labels=video_info['annotations'] 
        vlabels = self.cate_dict[name]
        for j in range(len(video_labels)):
            tmp_info=video_labels[j]
            tmp_start=tmp_info['segment'][0]
            tmp_end=tmp_info['segment'][1]
            tmp_start=max(min(1,tmp_start/corrected_second),0)
            tmp_end=max(min(1,tmp_end/corrected_second),0)
            tmp_fea_start = tmp_start*self.x_tdim
            tmp_fea_end = tmp_end*self.x_tdim
            s_bbox.append([tmp_fea_start,tmp_fea_end])
        for kk in range(len(s_bbox),self.gt_hold_num):
            s_bbox.append([-1.0,-1.0])
        tmp_df=pd.read_csv(self.csv_dir+"/"+name+".csv",skiprows=1,header=None,sep=',')
        anchor_feature = tmp_df.values[:,:]
        result.append(anchor_feature)
        result.append(s_bbox)
        result.append(name)
        result.append(vlabels)
        result.append(video_second)
        return result
    def load_queue(self,start,end):
        bindex = 0
        lendata = end - start
        while True:
            t = self.load_single_feature(bindex+start)
            self.queue.put(t)
            bindex += 1
            if bindex == lendata:
                bindex = 0
    def gen(self):
        while True:
            t = self.queue.get()
            yield t[0],t[1],t[2],t[3],t[4] 
    def stop_process(self):
        for t in self.process_array:
            t.terminate()
            t.join()
        print('Stop the process sucess.')

#-------------------------- Test Data Loader -----------------------------------#
class KineticsDatasetLoadTrimList():
    def __init__(self,dataSet,csv_dir,csv_oris_dir,gt_hold_num,x_tdim,df_file='./cvs_records_k600/k600_train_val_info.csv',resize_dim=256):
        train_video_dict, val_video_dict, test_video_dict=getDatasetDictK600(df_file=df_file)
        if "train" in dataSet:
            self.train_dict=train_video_dict
        else:
            self.train_dict=val_video_dict
        self.dataSet = dataSet
        self.resize_dim = resize_dim
        self.train_list = list(self.train_dict.keys())
        self.train_list_copy = self.train_list 
        print('The %s video num is: %d'%(self.dataSet,len(self.train_list)))
        print('Load k600 train val gt info...')
        cate_info = pd.read_csv('k600_train_val_gt.txt',sep = ' ',header = None)
        self.cate_dict = {}
        for v_id in range(len(cate_info)):
            v = cate_info.loc[v_id][0]
            cate = cate_info.loc[v_id][2]
            self.cate_dict[v] = cate  
        print('Load Done.')
        self.shuffle()
        self.csv_dir = csv_dir
        self.csv_oris_dir = csv_oris_dir
        self.gt_hold_num = gt_hold_num
        self.x_tdim = x_tdim
        self.num_samples = len(self.train_list)
        self.batch_size = 1
        self.train_feature_list = multiprocessing.Manager().list() 
        self.process_num = 32
        self.process_array = []
        self.start_train_idx = 0
        for pid in range(self.process_num):
            pid_num =  int(self.num_samples / self.process_num)
            start_idx = pid*pid_num
            end_idx = (pid+1)*pid_num
            if pid == (self.process_num - 1):
                if end_idx < self.num_samples: end_idx = self.num_samples
            t = Process(target=self.load_queue,args=(start_idx,end_idx))
            self.process_array.append(t)
            t.start()
    def shuffle(self):
        randperm = np.random.permutation(len(self.train_list))
        self.train_list = [self.train_list_copy[int(randperm[idx])] for idx in range(len(randperm))]
    def load_single_feature(self,start):
        result = []
        name = self.train_list[start]
        s_bbox = []
        r_bbox = []
        label_bbox=[]
        slabel_box = []
        rlabel_bbox = []
        video_info=self.train_dict[name]
        video_frame=video_info['duration_frame']
        video_second=video_info['duration_second']
        feature_frame=video_info['feature_frame']
        feature_num = int(feature_frame / 8)
        corrected_second=float(feature_frame)/video_frame*video_second
        video_labels=video_info['annotations'] 
        vlabels = self.cate_dict[name]
        for j in range(len(video_labels)):
            tmp_info=video_labels[j]
            tmp_start=tmp_info['segment'][0]
            tmp_end=tmp_info['segment'][1]
            tmp_label=tmp_info['label']
            tmp_start=max(min(1,tmp_start/corrected_second),0)
            tmp_end=max(min(1,tmp_end/corrected_second),0)
            tmp_fea_start = tmp_start*self.x_tdim
            tmp_fea_end = tmp_end*self.x_tdim
            s_bbox.append([tmp_fea_start,tmp_fea_end])
            tmp_len = tmp_end - tmp_start
            if tmp_len*feature_num >= 8: 
                r_bbox.append([tmp_start,tmp_end])
        if  self.dataSet == 'val_moment': 
            tmp_df=pd.read_csv(self.csv_dir+"/"+name+".csv",skiprows=1,header=None,sep=',')
            anchor_feature = tmp_df.values[:,:]  
            for kk in range(len(s_bbox),self.gt_hold_num):
                s_bbox.append([-1.0,-1.0])         
            result.append(anchor_feature)
            result.append(s_bbox)
            result.append(name)
            result.append(vlabels)
            result.append(video_second)
            return result
        else:
            tmp_df=pd.read_csv(self.csv_oris_dir+"/"+name+".csv",skiprows=1,header=None,sep=',')
            tmp_values = tmp_df.values[:,:]
            for m in range(len(r_bbox)):
                tmp_start_fea,tmp_end_fea = int(r_bbox[m][0]*feature_num),int(r_bbox[m][1]*feature_num)
                proposal_fea = tmp_values[tmp_start_fea:tmp_end_fea,:]
                proposal_duration,proposal_frame = (r_bbox[m][1]-r_bbox[m][0])*video_second,(r_bbox[m][1]-r_bbox[m][0])*video_frame
                proposal_inter_fea = pool_fea_interpolate_th(proposal_fea,len(proposal_fea),proposal_duration,proposal_frame,self.resize_dim)
                result.append([proposal_inter_fea,[[128.0,384.0]],name,vlabels,proposal_duration*2])        
            return result
    def load_queue(self,start,end):
        bindex = 0
        lendata = end - start
        for i in range(lendata):
            t = self.load_single_feature(bindex+start)
            if self.dataSet == 'val_moment':
                self.train_feature_list.append(t)
            else:
                for z in range(len(t)):
                    self.train_feature_list.append(t[z])
            bindex += 1
            if bindex == lendata: bindex = 0
    def gen(self):
        self.num_samples = len(self.train_feature_list)
        print('The sample number of generator %s is %d'%(self.dataSet,self.num_samples))
        while True:
            t = self.train_feature_list[self.start_train_idx]
            self.start_train_idx += 1
            if self.start_train_idx == self.num_samples:
                self.start_train_idx = 0 
            yield t[0],t[1],t[2],t[3],t[4] 
    def stop_process(self):
        for t in self.process_array:
            t.terminate()
            t.join()
        print('Stop the process sucess.')
