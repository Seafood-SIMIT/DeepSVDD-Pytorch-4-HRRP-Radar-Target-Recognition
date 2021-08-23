
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_net import BaseNet

def build_autoencoder():
    ae_net = DeepAutoEncoder()
    return ae_net


class DeepAutoEncoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 8
        
        # 二维图转为一维雷达高分辨距离像
        self.conv1 = nn.Conv1d(in_channels=1,out_channels = 32, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32,out_channels = 16, kernel_size=2)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(in_channels=16,out_channels = 8, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(8)
        self.fc1 = nn.Linear(8*31, self.rep_dim, bias=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose1d(1,8,kernel_size = 1)
        self.bn11 = nn.BatchNorm1d(8)
        self.deconv2 = nn.ConvTranspose1d(8,16,kernel_size = 1)
        self.bn12 = nn.BatchNorm1d(16)
        self.deconv3 = nn.ConvTranspose1d(16, 32, kernel_size=1)
        self.bn13 = nn.BatchNorm1d(32)
        self.deconv4 = nn.ConvTranspose1d(32, 1, kernel_size=1)


    def forward(self, x):
        x = x.unsqueeze(1)      #[batch,1,20]
        x = self.conv1(x)       #[batch,32,20]
        x = self.conv2(x)       #[batch,8,19]
        x = self.conv3(x)       #[batch,8,19]
        x = x.view(x.size(0), -1)       #[batch,8*19]
        #print(x.shape)
        x = self.fc1(x)                 #[batch,8]
        x = x.view(x.size(0), int(self.rep_dim / 8),8) #[batch,2,4]
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)  #[batch,2,16]
        x = self.deconv1(x)     #[batch,8,16]
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)  #[batch,2,32]
        x = self.deconv2(x)     #[batch,32,32]
        x = self.deconv3(x)     #[batch,1,32]
        x = self.deconv4(x)     #[batch,1,32]
        x = torch.sigmoid(x)

        return x