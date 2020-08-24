# AherNet: Learning to Localize Actions from Moments

### Environment

These codes are implemented in TensorFlow (1.12.3) GPU version. 
The version of operation system in docker is Ubuntu16.04.
The python version is 3.6.7. 
The GPU is NVIDIA Tesla P40 (23GB) and the CUDA version is CUDA 9.0, cudnn version is 7.3.1

### Note
This repository includes all the pure codes and configuration files of the final setting in the paper (From ActivityNet v1.3 to Kinetics-600, i.e., ANet -> K600).

All the training/validation features and related information files of the dataset are not included in the repository since they are too large, which will be released on google drive or Baidu Yun in the future. 

The file `k600_val_annotation.txt` contains the mannul temporal annotations of 6,459 videos in the validation set of Kinetics-600.


### Training of AherNet

The training script is `aher_k600_train.py`. 
Once you have all the training features and information files, you could run

```
CUDA_VISIBLE_DEVICES=0 python3 aher_k600_train.py
```

### Testing of AherNet on Kinetics-600

The testing script is `aher_k600_test.py`. 
Once you have all the testing features and information files, you could run

```
CUDA_VISIBLE_DEVICES=0 python3 aher_k600_test.py
```
You can evaluate several snapshots (models) during testing stage to find which model is the best one.

After the evaluation, you can get the results of ".json" file which will be evaluated by `./evaluation/get_proposal_performance.py` or `./evaluation/get_detection_performance.py` for temporal action proposal and temporal action localization evaluation.


### Citation

If you use these models in your research, please cite:

    @inproceedings{Long:ECCV20,
      title={Learning To Localize Actions from Moments},
      author={Fuchen Long, Ting Yao, Zhaofan Qiu, Xinmei Tian, Jiebo Luo and Tao Mei},
      booktitle={ECCV},
      year={2020}
    }
	