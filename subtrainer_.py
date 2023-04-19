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



#-----------------------------------------------------projecttrain-----------------------------------------------------------
def train_Sin(it,net,train_iter,valid_iter,num_gpus,num_epochs,LR,save_name = None): 
    print('-------------------------{}th Iteration Sin Subnet Training had Started-------------------------'.format(it+1))
    devices = [torch.device('cuda:'+str(i)) for i in range(num_gpus)]
    net = nn.DataParallel(net,device_ids=devices).to(devices[0])
    loss = nn.MSELoss()
    rmse_make = []
    best_rmse = 100
    for epoch in range(num_epochs):
        if it < 3:
            optimizer = torch.optim.Adam(net.parameters(),lr=LR[0])
            LearningRate = LR[0]
        else:
            if epoch < 50:
                optimizer = torch.optim.Adam(net.parameters(),lr=LR[0])
                LearningRate = LR[0]
            elif epoch <70:
                optimizer = torch.optim.Adam(net.parameters(),lr=LR[1])
                LearningRate = LR[1]
            else:
                optimizer = torch.optim.Adam(net.parameters(),lr=LR[2])
                LearningRate = LR[2]
        net.train()
        train_loss,train_psnr,train_ssim,train_rmse = [],[],[],[]
        for i,(input_,target_) in enumerate(train_iter):
            optimizer.zero_grad()
            
            if input_.shape[1] != 1: # patch_size
                patch_size = input_.shape[2]
                input_ = input_.view(-1,1,patch_size,patch_size) 
                target_ = target_.view(-1,1,patch_size,patch_size)

            input_,target_ = input_.to(devices[0],dtype=torch.float),target_.to(devices[0],dtype=torch.float)
            pred_ = net(input_)
            l = loss(pred_,target_)
            l.backward()
            optimizer.step()
            
            train_loss.append(l)
            
        train_loss = sum(train_loss) / len(train_loss)

        # 训练时只需要盯 loss 就行，不需要看评价指标
        print('epoch: {},LearningRate：{},train_loss: {:.6f}'.format(epoch+1,LearningRate,train_loss))
        # valid
        net.eval()
        valid_psnr,valid_ssim,valid_rmse = [],[],[]
        
        for i,(input_,target_) in enumerate(valid_iter):

            if input_.shape[1] != 1: # patch_size
                patch_size = input_.shape[2]
                input_ = input_.view(-1,1,patch_size,patch_size) 
                target_ = target_.view(-1,1,patch_size,patch_size)
            

            input_,target_ = input_.to(devices[0],dtype=torch.float),target_.to(devices[0],dtype=torch.float)
            with torch.no_grad():      # 验证的时候不用计算梯度，取消掉它减少时间和空间的损耗
                pred_ = net(input_)


            data_range = pred_.max() if pred_.max() > target_.max() else target_.max()
            psnr,ssim,rmse = compute_measure(pred_,target_,data_range=data_range) 

            valid_rmse.append(rmse)
            

        valid_rmse = sum(valid_rmse) / len(valid_rmse)
        rmse_make.append(valid_rmse)
        
        print('\t \t valid_rmse: {:.6f}'
              .format(valid_rmse))
        
        if best_rmse > valid_rmse:
            best_rmse = valid_rmse
            print('best rmse update: {:.6f}'.format(best_rmse))
            if save_name:
                torch.save(net, save_name)

                print('saving model with rmse {:.6f}'.format(best_rmse))
        print('---------------------------------------------------')
    np.save('./2016_model_save/128views/Sin_rmse_it_{}.npy'.format(it),np.array(rmse_make))
    print('-------------------------{}th Iteration Sin Subnet Training had Finished-------------------------'.format(it+1))



#-------------------------------------------------------imagetrain-------------------------------------------------------
def train_Img(it,net,train_iter,valid_iter,num_gpus,num_epochs,LR,save_name = None): 
    print('-------------------------{}th Iteration Img Subnet Training had Started-------------------------'.format(it+1))
    devices = [torch.device('cuda:'+str(i)) for i in range(num_gpus)]
    net = nn.DataParallel(net,device_ids=devices).to(devices[0])
    loss = nn.MSELoss()
    rmse_make = []
    best_psnr = 0  
    for epoch in range(num_epochs):
        if it < 4:
            optimizer = torch.optim.Adam(net.parameters(),lr=LR[0])
            LearningRate = LR[0]
        else:
            if epoch < 350:
                optimizer = torch.optim.Adam(net.parameters(),lr=LR[0])
                LearningRate = LR[0]
            else:
                optimizer = torch.optim.Adam(net.parameters(),lr=LR[1])
                LearningRate = LR[1]
        # train
        net.train()
        train_loss,train_psnr,train_ssim,train_rmse = [],[],[],[]
        for i,(input_,target_) in enumerate(train_iter):
            optimizer.zero_grad()
            
            # if input_.shape[1] != 1: # patch_size
            #     patch_size = input_.shape[2]
            #     input_ = input_.view(-1,1,patch_size,patch_size) 
            #     target_ = target_.view(-1,1,patch_size,patch_size)
            input_,target_ = input_.to(devices[0],dtype=torch.float),target_.to(devices[0],dtype=torch.float)
            pred_ = net(input_)
            l = loss(pred_,target_)
            l.backward()
            optimizer.step()
            
            train_loss.append(l)
            
        train_loss = sum(train_loss) / len(train_loss)

        print('epoch: {},LearningRate：{},train_loss: {:.6f}'.format(epoch,LearningRate,train_loss))
        # valid
        net.eval()
        valid_psnr,valid_ssim,valid_rmse = [],[],[]
        
        for i,(input_,target_) in enumerate(valid_iter):

            # if input_.shape[1] != 1: # patch_size
            #     patch_size = input_.shape[2]
            #     input_ = input_.view(-1,1,patch_size,patch_size) 
            #     target_ = target_.view(-1,1,patch_size,patch_size)
            

            input_,target_ = input_.to(devices[0],dtype=torch.float),target_.to(devices[0],dtype=torch.float)
            with torch.no_grad():      
                pred_ = net(input_)

            shape_ = input_.shape[-1]
            input_ = decoder(input_.view(9,shape_, shape_)) # (batch_size,9,512,512)
            target_ = decoder(target_.view(9,shape_, shape_))
            pred_ = decoder(pred_.view(9,shape_, shape_))
            data_range = pred_.max() if pred_.max() > target_.max() else target_.max()
            psnr,ssim,rmse = compute_measure(pred_,target_,data_range=data_range) 
            valid_psnr.append(psnr)
            valid_ssim.append(ssim)
            valid_rmse.append(rmse)
            
        valid_psnr = sum(valid_psnr) / len(valid_psnr)
        valid_ssim = sum(valid_ssim) / len(valid_ssim)
        valid_rmse = sum(valid_rmse) / len(valid_rmse)
        rmse_make.append(valid_rmse)
        print('\t valid_psnr: {:.4f}, valid_ssim: {:.4f}, valid_rmse: {:.6f}'
              .format(valid_psnr,valid_ssim,valid_rmse))
        
        if best_psnr < valid_psnr:
            best_psnr = valid_psnr
            print('best psnr update: {:.4f}'.format(best_psnr))
            if save_name:
                torch.save(net, save_name)
                # # 加载
                # model = torch.load('\model.pkl')
                print('saving model with psnr {:.4f}'.format(best_psnr))
        print('-----------------------------------------------------------------------')
    print('-------------------------{}th Iteration Img Subnet Training had Finished-------------------------'.format(it+1))
    print('----------------------------------------------------------------')
    print('|This Iteration Img Subnet Performance(Validation PSNR)：{:.4f}|'.format(best_psnr))
    print('----------------------------------------------------------------')
    
def decoder(num):# input 9 channel, elements just ‘0 1’ output 1 channel
    #num[num == 255] = 1
    #num = num.to('cpu').numpy()
    num = num.to('cpu')
    num = torch.round(num)
    img_1c = torch.zeros(512,512)
    for k in range(9):
        img_1c= (img_1c + num[k,:,:]*(2 ** k))
    return torch.Tensor(img_1c).to('cuda:0')