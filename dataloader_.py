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



class ct_dataset(Dataset):
    def __init__(self,data_path,patientsID,is_firstIt,D,is_upsample=False):
        super().__init__()
        if D == 0:
            if is_firstIt == 0:
                input_path = sorted(glob(os.path.join(data_path,'*_input_sv.npy')))
                target_path = sorted(glob(os.path.join(data_path,'*_target.npy')))
            else:
                input_path = sorted(glob(os.path.join(data_path,'*_input_mid.npy')))
                target_path = sorted(glob(os.path.join(data_path,'*_target.npy')))
        else:
            input_path = sorted(glob(os.path.join(data_path,'*_input_mid.npy')))
            target_path = sorted(glob(os.path.join(data_path,'*_target.npy')))
        self.input_ = [f for f in input_path if re.search(r"(C|N|L)\d{3}", f)[0] in patientsID]
        self.target_ = [f for f in target_path if re.search(r"(C|N|L)\d{3}", f)[0] in patientsID]
    def __len__(self):
        return len(self.input_)
    def __getitem__(self, idx):

        input_img, target_img = self.input_[idx], self.target_[idx]
        input_img, target_img = np.load(input_img),np.load(target_img)
        input_img, target_img =torch.Tensor(input_img),torch.Tensor(target_img)

        return input_img.unsqueeze(0),target_img.unsqueeze(0)
    
def get_loader(it,D,data_path=None,patientsID_path=None,batch_size=32,num_workers=8,is_upsample=False):
    train_patientsID = np.load(os.path.join(patientsID_path,'train_patientsID.npy')).flatten()
    valid_patientsID = np.load(os.path.join(patientsID_path,'valid_patientsID.npy')).flatten()
    test_patientsID  = np.load(os.path.join(patientsID_path,'test_patientsID.npy')).flatten()

    train_dataset_ = ct_dataset(data_path,train_patientsID,is_firstIt = it,D=D,is_upsample=is_upsample)
    valid_dataset_ = ct_dataset(data_path,valid_patientsID,is_firstIt = it,D=D,is_upsample=is_upsample)
    test_dataset_  = ct_dataset(data_path,test_patientsID,is_firstIt = it,D=D,is_upsample=is_upsample)
    #print('train_dataset numbers:{}\nvalid_dataset numbers:{}\ntest_dataset numbers:{}\n'.format(len(train_dataset_),len(valid_dataset_),len(test_dataset_)))
          
    return (DataLoader(dataset = train_dataset_,batch_size=batch_size,shuffle=False,num_workers=num_workers),
            DataLoader(dataset = valid_dataset_,batch_size=1,shuffle=False,num_workers=num_workers),
            DataLoader(dataset = test_dataset_,batch_size=batch_size,shuffle=False,num_workers=num_workers) )

