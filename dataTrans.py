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



def DataTrans(save_name,ProjectPath,ImagePath,tag,it):
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    #load net
    net = torch.load(save_name)
    net = net.eval()
    # define operator
    d = torch.load(os.path.join(config.RESULTS_PATH,"operator_radon_bwd_train_phase_1","model_weights.pt",),map_location=device,)
    radon_net = RadonNet.new_from_state_dict(d)
    # overwrite default settings (if necessary)
    radon_net.OpR.flat = True
    radon_net.OpR.filter_type = "hamming"
    operator = radon_net.OpR.to(device)
    #print(list(operator.parameters()))
    radon_net.freeze()
    #parameters setting
    radon = FanbeamRadon(n = [256, 256],
        angles = torch.linspace(0, 360, 513, requires_grad=False)[:-1],
        scale = torch.tensor(0.01, requires_grad=False),
        d_source = torch.tensor(1000.00, requires_grad=False),
        n_detect = 1024,
        s_detect = torch.tensor(-1.0, requires_grad=False),
        flat=True,
        filter_type="hamming",
        learn_inv_scale=False)
    radon = radon.to(device)
    
    
    if tag == 1:
        print('-------------------------{}th Iteration Sin→Img Data Trans had Started-------------------------'.format(it+1))
        print('------------------------------------------ Transforming... ------------------------------------')
        path_ = sorted(os.listdir(ProjectPath))
        data_ = []
        data = []
        save_path = ImagePath
        if it == 0:
            for i in range(len(path_)):
                if 'input_sv' in path_[i]:#第一轮迭代匹配字段为input_sv
                    data.append(path_[i])#data = [2378张正弦input的地址]
        else:
            for i in range(len(path_)):
                if 'input_mid' in path_[i]:#第二轮开始为input_mid
                    data.append(path_[i])#data = [2378张正弦input的地址]
        for i in range(len(data)):
            data_.append(os.path.join(ProjectPath,data[i]))
        for index in range(len(data)):
            inp_ = np.load(data_[index])
            inp = torch.Tensor(inp_).unsqueeze(0).unsqueeze(0).to(device)#正弦输入
            with torch.no_grad():
                tar_ = net(inp)#正弦输出
            tar = radon.inv(tar_).squeeze(0).squeeze(0)#利用反演算子将net获取到的正弦转到图像
            str_index = data[index].find('Sin_input')#索引到原np的病例号和编号 
            name_start = data[index][0:str_index]#前缀名儿：病例号_编号_
            im_inp_name = name_start + 'Img_input_mid.npy'
            en = tar.cpu().numpy()
            en = (en - en.min())/(en.max()-en.min())
            tar_save = encoder(en)
            np.save(os.path.join(save_path,im_inp_name),tar_save)#将转换到的图像存储为ImagePairs目录下的input
        print('-------------------------{}th Iteration Sin→Img Data Trans had Finished-------------------------'.format(it+1))
    else:
        print('-------------------------{}th Iteration Data had Started To Generating..-------------------------'.format(it+2))
        print('---------------------------------------- Generating...-------------------------------------------')
        path_ = sorted(os.listdir(ImagePath))
        data_ = []
        data = []
        save_path = ProjectPath
        for i in range(len(path_)):
            if 'input_mid' in path_[i]:
                data.append(path_[i])#data = [2378张图像input的地址]
        for i in range(len(data)):
            data_.append(os.path.join(ImagePath,data[i]))
        for index in range(len(data_)):
            inp_ = np.load(data_[index])#load一张图像
            inp = torch.Tensor(inp_).unsqueeze(0).unsqueeze(0).to(device)#图像输入（1，1，512，512）
            with torch.no_grad():
                tar_ = net(inp)#图像输出（1，1，512，512）
            de = tar_.squeeze(0).squeeze(0).cpu()
            de = decoder(de).unsqueeze(0).unsqueeze(0).to(device)
            tar = radon(de).squeeze(0).squeeze(0)#利用正演算子将net获取到的图像转到正弦（512，1024）
            str_index = data[index].find('Img_input')#索引到原np的病例号和编号
            name_start = data[index][0:str_index]#前缀名儿：病例号_编号_
            im_inp_name = name_start + 'Sin_input_mid.npy'#这里我将第一轮迭代生成的正弦np文件名后面加了一个下划线 这样可以不覆盖原有的稀疏视图数据
            np.save(os.path.join(save_path,im_inp_name),tar.cpu().numpy())#将转换到的存储为ProjectPairs目录下的input
        print('-------------------------{}th Iteration Data Generation Had Been Done！！-------------------------'.format(it+2))

def encoder(num):
    img_8c = np.zeros([8,256,256])
    num = num.astype(int)
    for i in range(256):
        for j in range(256):
            num_b = bin(num[i,j])
            lenth = len(num_b)
            for k in range(lenth):
                if num_b[lenth-1-k] == '1':
                    img_8c[k,i,j] = 1
    img_8c = img_8c.astype(float)
    return img_8c   
    
def decoder(num):# input 8 channel, elements just ‘0 1’ output 1 channel
    #num[num == 255] = 1
    #num = num.to('cpu').numpy()
    num = num.to('cpu')
    num = torch.round(num)
    img_1c = torch.zeros(256,256)
    for k in range(8):
        img_1c= (img_1c + num[k,:,:]*(2 ** k))
    return torch.Tensor(img_1c).to('cuda:0')
        
        
        
        
        
