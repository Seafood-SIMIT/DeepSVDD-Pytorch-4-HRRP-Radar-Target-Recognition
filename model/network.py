import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_net import BaseNet

class DeepSVDDNetwork(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 8
        
        # 二维图转为一维雷达高分辨距离像
        self.conv1 = nn.Conv1d(in_channels=1,out_channels = 32, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32,out_channels = 16, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(in_channels=16,out_channels = 8, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(8)
        self.fc1 = nn.Linear(8*32, self.rep_dim, bias=False)


    def forward(self, x):
        x = x.unsqueeze(1)      #[batch,1,20]
        x = self.conv1(x)       #[batch,32,20]
        x = self.conv2(x)       #[batch,8,19]
        x = self.conv3(x)       #[batch,8,19]
        x = x.view(x.size(0), -1)       #[batch,8*19]
        #print(x.shape)
        x = self.fc1(x)                 #[batch,8]

        return x