
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

        self.rep_dim = 32
        #self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        # 二维图转为一维雷达高分辨距离像
        self.conv1 = nn.Conv1d(in_channels=1,out_channels = 533, kernel_size=2)
        #self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        #self.conv2 = nn.Conv1d(in_channels=1,out_channels = 533, kernel_size=2)
        #self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        #self.fc1 = nn.Linear(533*532, self.rep_dim, bias=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose1d(533,1,kernel_size = 2)
        #self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        #self.deconv2 = nn.ConvTranspose1d(4, 8, 5, bias=False, padding=3)
        #self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        #self.deconv3 = nn.ConvTranspose1d(8, 1, 5, bias=False, padding=2)

    def forward(self, x):
        x = x.unsqueeze(1)      #[batch,533]
        x = self.conv1(x)       #[batch,533,532]
        #x = self.pool(F.leaky_relu(self.bn1(x)))
        #x = self.conv2(x)
        #x = self.pool(F.leaky_relu(self.bn2(x)))
        #x = x.view(x.size(0), -1)
        #x = self.fc1(x)
        #x = x.view(x.size(0), int(self.rep_dim / 16), 4, 4)
        #x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)     #[batch,32,533]
        #print(x.size())
        #x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        #x = self.deconv2(x)
        #x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        #x = self.deconv3(x)
        x = torch.sigmoid(x)

        return x