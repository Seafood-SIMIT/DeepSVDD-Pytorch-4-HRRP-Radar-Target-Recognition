
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_net import BaseNet

def build_autoencoder():
    ae_net = DeepAutoEncoder()
    return ae_net

class Interpolate(nn.Module):
    def __init__(self, scale_factor=2):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor)
        return x

class DeepAutoEncoder(BaseNet):

    def __init__(self):
        super().__init__()
        self.rep_dim = 8

        self.inter_layer = Interpolate(scale_factor=2)
        self.conv = nn.Sequential(
            ##cnn1
            nn.Conv1d(in_channels=1,out_channels = 8, kernel_size=5, bias=False, padding=2),
            nn.BatchNorm1d(8,eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool1d(2,2),
            nn.Conv1d(in_channels=8,out_channels = 4, kernel_size=5, bias=False, padding=2),
            nn.BatchNorm1d(4,eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool1d(2,2)
        )

        self.fc = nn.Sequential(
            nn.Linear(4*8, self.rep_dim, bias=False)
        )

        self.deconv = nn.Sequential(
            nn.LeakyReLU(),
            self.inter_layer,
            nn.ConvTranspose1d(2,4,kernel_size = 5, bias=False, padding=2),
            nn.BatchNorm1d(4,eps=1e-04, affine=False),
            nn.LeakyReLU(),
            self.inter_layer,
            nn.ConvTranspose1d(4,8,kernel_size = 5, bias=False, padding=2),
            nn.BatchNorm1d(8,eps=1e-04, affine=False),
            nn.LeakyReLU(),
            self.inter_layer,
            nn.ConvTranspose1d(8, 1, kernel_size=5, bias=False, padding=2),
        )
        # Decoder
        #self.bn13 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = x.unsqueeze(1)      #[batch,1,32]
        x = self.conv(x)       #[batch,32,32]
        #x, (h_n,c_n) = self.lstm(x)     #[batch,input,hidden]
        #print(x.shape)
        x = x.contiguous().view(x.size(0), -1)       #[batch,input*hidden]
        #print(x.shape)
        x = self.fc(x)                 #[batch,rep_dim]
        #x = x.unsqueeze(1)      #[batch,1,repdim]
        #print(x.shape)
        x = x.view(x.size(0), int(self.rep_dim/4),4) #[batch,2,8]
        #print(x.shape)
        #x = F.interpolate(F.leaky_relu(x),scale_factor=4)
        x = self.deconv(x)     #[batch,8,16]
        x = x.view(x.size(0),-1)
        #print(x.shape)
        x = torch.sigmoid(x)
        return x