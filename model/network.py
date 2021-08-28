import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_net import BaseNet

class DeepSVDDNetwork(BaseNet):

    def __init__(self):
        super().__init__()


        self.rep_dim = 8

        self.R = torch.nn.Parameter(torch.tensor(0.5,requires_grad = True))  # radius R initialized with 0 by default.
        self.c = torch.nn.Parameter(torch.zeros((1,self.rep_dim),requires_grad=True))
        self.register_parameter("Radius",self.R)
        self.register_parameter("c",self.c)

        self.pool = nn.MaxPool1d(2, 2)
        
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
        # self.lstm = nn.LSTM(
        #                     input_size=32,
        #                     hidden_size=16,
        #                     batch_first=True,
        #                     bidirectional=False,
        #                     )



    def forward(self, x):
        x = x.unsqueeze(1)      #[batch,1,32]
        x = self.conv(x)       #[batch,out,32]
        #x = self.pool(F.leaky_relu(x))
        #x, (h_n,c_n) = self.lstm(x)     #[batch,input,hidden]
        #print(x.shape)
        x = x.contiguous().view(x.size(0), -1)       #[batch,input*hidden]
        #print(x.shape)
        x = self.fc(x)                 #[batch,rep_dim]
        #x = torch.tanh(x)

        return x