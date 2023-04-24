import matplotlib.pyplot as plt
from networks import RadonNet,FanbeamRadon
from operators import l2_error
import config 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import re
from glob import glob 
from measure import compute_measure
from torch.utils.data import Dataset,DataLoader

from operators import EnhancedAlignedLoss
from subtrainer_ import train_Sin,train_Img
from dataTrans import DataTrans
from networks import Unet_sin,Unet_img
from dataloader_ import ct_dataset


It_nums = 6
LR = [1e-4,5e-5,2e-5]
Sin_path = '/dataset/2020AAPM/128views/ProjectPairs'
Img_path = '/dataset/2020AAPM/128views/ImagePairs'
patientID = './2020_mayo_patientsID'
num_epochs_Sin = [200,100,100,100,100,100]
num_epochs_Img = [200,100,100,100,100,100]
num_gpus = 2
batch_size = [20,36]
num_workers = 8
net1 = Unet_sin()
net2 = Unet_img()
for it in range(It_nums):

    data_path = Sin_path
    D = 0
    train_iter,valid_iter,_= get_loader(it,D,data_path,patientID,batch_size[0],num_workers)
    save_name = './2020_model_save/128views/NetPD_Iteration-0{}.pkl'.format(it+1)
    train_Sin(it,net,train_iter,valid_iter,num_gpus,num_epochs_Sin[it],LR,save_name=save_name)
    torch.cuda.empty_cache()

    tag = 1
    DataTrans(save_name,Sin_path,Img_path,tag,it)
    psnr_mid_calculate(Img_path)
    torch.cuda.empty_cache()

    data_path = Img_path
    D = 1
    _train_iter,_valid_iter,s_= get_loader(it,D,data_path,patientID,batch_size[1],num_workers)
    save_name = './2020_model_save/128views/NetID_Iteration-0{}.pkl'.format(it+1)
    train_Img(it,net,_train_iter,_valid_iter,num_gpus,num_epochs_Img[it],LR,save_name=save_name)
    torch.cuda.empty_cache()
    tag = 0
    DataTrans(save_name,Sin_path,Img_path,tag,it)
    torch.cuda.empty_cache()
