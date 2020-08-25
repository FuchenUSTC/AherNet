# AherNet: Learning to Localize Actions from Moments

This repository includes the codes and configuration files of the transfer setting of "From ActivityNet v1.3 to Kinetics-600, i.e., ANet -> K600" in the paper.

All the training/validation features and related meta files of the dataset are not uploaded in the repository since they are too large, which will be released on google drive or Baidu Yun in the future. 

The file `k600_val_annotation.txt` contains the mannul temporal annotations of 6,459 videos in the validation set of Kinetics-600.

# Update
* 2020.8.25: Repository for AherNet and annotations of sampled validation video in Kinetics-600

# Contents:

* [Paper Introduction](#paper-introduction)
* [Environment](#environment)
* [Training of AherNet](#training-of-ahernet)
* [Testing of AherNet on Kinetics-600](#testing-of-ahernet-on-kinetics-600)
* [Citation](#citation)

# Paper Introduction

<img src="https://github.com/FuchenUSTC/AherNet/raw/master/pic/eccv_framework.JPG" width="700" alt="image" align=center />

With the knowledge of action moments (i.e., trimmed video clips that each contains an action instance), humans could routinely localize an action temporally in an untrimmed video. Nevertheless, most practical methods still require all training videos to be labeled with temporal annotations (action category and temporal boundary) and develop the models in a fully-supervised manner, despite expensive labeling efforts and inapplicable to new categories. In this paper, we introduce a new design of transfer learning type to learn action localization for a large set of action categories, but only on action moments from the categories of interest and temporal annotations of untrimmed videos from a small set of action classes. Specifically, we present Action Herald Networks (AherNet) that integrate such design into an one-stage action localization framework. Technically, a weight transfer function is uniquely devised to build the transformation between classification of action moments or foreground video segments and action localization in synthetic contextual moments or untrimmed videos. The context of each moment is learnt through the adversarial mechanism to differentiate the generated features from those of background in untrimmed videos. Extensive experiments are conducted on the learning both across the splits of ActivityNet v1.3 and from THUMOS14 to ActivityNet v1.3. Our AherNet demonstrates the superiority even comparing to most fully-supervised action localization methods. More remarkably, we train AherNet to localize actions from 600 categories on the leverage of action moments in Kinetics-600 and temporal annotations from 200 classes in ActivityNet v1.3.

# Environment

TensorFlow version: 1.12.3 of GPU

Operation system in docker version: Ubuntu16.04

Python version: 3.6.7

GPU version: NVIDIA Tesla P40 (23GB)

Cuda and Cudnn version: CUDA 9.0 and cudnn 7.3.1


# Training of AherNet

The training script is `aher_k600_train.py`. 
Once you have all the training features and information files, you could run

```
CUDA_VISIBLE_DEVICES=0 python3 aher_k600_train.py
```

# Testing of AherNet on Kinetics-600

The testing script is `aher_k600_test.py`. 
Once you have all the testing features and information files, you could run

```
CUDA_VISIBLE_DEVICES=0 python3 aher_k600_test.py
```
You can evaluate several snapshots (models) during testing stage to find which model is the best one.

After the evaluation, you can get the results of ".json" file which will be evaluated by `./evaluation/get_proposal_performance.py` or `./evaluation/get_detection_performance.py` for temporal action proposal and temporal action localization evaluation.


# Citation

If you use these models in your research, please cite:

    @inproceedings{Long:ECCV20,
      title={Learning To Localize Actions from Moments},
      author={Fuchen Long, Ting Yao, Zhaofan Qiu, Xinmei Tian, Jiebo Luo and Tao Mei},
      booktitle={ECCV},
      year={2020}
    }
	