
import os
import glob
import torch

from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import scipy.io as scio #读取mat文件使用的python库
from itertools import compress
from sklearn.model_selection import train_test_split
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
def read_data(mat_list,hp):
    total_data = []
    total_label= []
    for now_image_name in mat_list:
        now_image_file = hp.data.train_dir + '/' + now_image_name   #序列文件
        now_gate_file = now_image_file.replace('.mat','_gate.mat')  #gate文件
        now_label_file = now_image_file.replace('.mat', '.txt') #标签文件
        radar_pulse_squence = scio.loadmat(now_image_file)   # 雷达回波读取
        radar_data_gate = scio.loadmat(now_gate_file) # 距离序列信息读取
    
        with open(now_label_file, "r") as f:
            data_value_str=f.readlines()

        data_value = []
        for i in range(len(data_value_str)):
            data_value.append(float(data_value_str[i].split()[3]))
        
        data_value = np.array(data_value)

        #abs操作
        radar_pulse_squence = abs(radar_pulse_squence['radar_pulse_sequence'])
        radar_gate_step = radar_data_gate['radar_data_gate_step']
        radar_gate_start = radar_data_gate['radar_data_gate_start'].reshape(1000)
    
        label=np.zeros((1000,1))
        ## positive data
        data_n = np.zeros((1000,hp.data.array_length))
        label_n = -1*np.ones((1000,1))
        data_p = np.zeros((1000*hp.data.ntp,hp.data.array_length))
        label_p = np.ones((1000*hp.data.ntp,1))
        for item in range(len(radar_pulse_squence)):
            #item_file = self.data_list[idx]
            # data
            #print(data_value[item],radar_gate_start[item],radar_gate_step)
            n_index = int((data_value[item] - radar_gate_start[item])/radar_gate_step.reshape(1)[0])
            #print(p_index[0])
            data_n[item,:] = radar_pulse_squence[item,(n_index-int(hp.data.array_length/2)):(n_index+int(hp.data.array_length/2))]
            data_n[item,:] = normalization(data_n[item,:])
            for i_p_random in range(hp.data.ntp):
                p_index = int(np.random.randint(low=hp.data.array_length/2,high=533-hp.data.array_length/2,size=1))
                if p_index in range((n_index-int(hp.data.array_length/2)),(n_index+int(hp.data.array_length/2))):
                    i_p_random = i_p_random -1
                    continue
                else:
                    data_p[item*hp.data.ntp+i_p_random,:] = radar_pulse_squence[item,(p_index-int(hp.data.array_length/2)):(p_index+int(hp.data.array_length/2))]
                    data_p[item*hp.data.ntp+i_p_random,:] = normalization(data_p[item*hp.data.ntp+i_p_random,:])

        total_data.append(data_p)
        total_data.append(data_n)
        total_label.append(label_p)
        total_label.append(label_n)
    return total_data,total_label
#制作dataloader
def create_dataloader(hp, logger, args):

    ImgFoldPath = hp.data.train_dir; # 平台输入数据固定目录，平台数据都存放在该目录下
    ResultFoldPath = '/data/result';  # 平台输出数据固定目录，输出结果都存放在该目录下
    FileList = os.listdir(ImgFoldPath)
    isImgfile = ['.mat' in i and 'gate' not in i for i in FileList] # 找到该目录下所有.mat文件
    ImgFileList = list(compress(FileList,isImgfile))                # 找到该目录下所有.mat文件

    data_read, label_read = read_data(ImgFileList,hp)
    #数据准备
    data = np.vstack((data_read))
    #print(data.shape)
    label = np.vstack((label_read))

    #显示大小
    logger.info("data total size: %s",data.shape)
    logger.info("label total size: %s",label.shape)
    #数据归一化
    #radar_pulse_squence = radar_pulse_squence/np.max(abs(radar_pulse_squence))
    #print(data_value.shape ,radar_gate_start.shape)
        
    train_set, test_set,train_label,test_label = train_test_split(data, label, test_size = hp.data.test_size,random_state = 1)
    
    #print(train_set.shape, test_set.shape)
    loader_train = DataLoader(dataset=HRRPDataset(hp, args, train_set,train_label),
                      batch_size = hp.ae.batch_size,
                      shuffle=True,
                      drop_last = True,
                      )
    loader_test = DataLoader(dataset=HRRPDataset(hp, args, test_set,test_label ),
                                   batch_size=1, shuffle=False, num_workers=0)
    return loader_train,loader_test
    
    
class HRRPDataset(Dataset):
    def __init__(self, hp, args, data_set,label_set):
        def find_all(file_format):
            return glob.glob(os.path.join(self.data_dir, file_format))
        self.hp = hp
        self.args = args
        self.radar_sequence = data_set
        self.label = label_set
        self.tensor_radar_sequence = torch.tensor(self.radar_sequence, dtype=torch.float,requires_grad = False)
        self.tensor_label = torch.tensor(self.label, dtype=torch.float)
        
        ##划分测试集
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return self.tensor_radar_sequence[idx], self.tensor_label[idx]
        #return 0, 0
            