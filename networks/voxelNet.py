from __future__ import print_function
import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys

import voxel_nin


class voxelNet(nn.Module):
    '''
        loading pretrained network and adding layers 
        Fine tuning and training the whole network
        computes a embedding given a voxel shape 
    '''

    def __init__(self, 3dnin_path, in_channels=1):
        super(voxelNet_pretrained, self).__init__()
        
        model = voxel_nin.voxel_nin
      
        model_path = 3dnin_path
        model.load_state_dict(torch.load(model_path))

        # remove layers
        model = nn.Sequential(*list(model.children())[:-4])


        #Input: 1@(30x30x30) 
        self.pretrain = nn.Sequential(*list(model.children()))
        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        
        self.drop1 = nn.Dropout(p=0.5)


    def forward(self, x):
        # returns the embedding

        batch_sz = x.size()[0]

        out = self.pretrain(x)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.drop1(out)
        out = self.fc3(out)

        # normalize the embedding
        out = F.normalize(out, p=2, dim=1)
        return out


class voxelNet_fixed(nn.Module):
    '''
        loading pretrained network and adding layers 
        Fine tuning and training only fc layers
        computes a similarity between two input voxel shapes
    '''

    def __init__(self, 3dnin_path, in_channels=1):
        super(voxelNet_pretrained_samePaper, self).__init__()
      
        model = voxel_nin.voxel_nin
      
        model_path = 3dnin_path
        model.load_state_dict(torch.load(model_path))

        # remove layers
        model = nn.Sequential(*list(model.children())[:-4])

        # set requires_grad to false to all the layers
        for name, module in model.named_children():
            for param in module.parameters():
                # print('child name %s , requires_grad %s' %(name, param.requires_grad))
                param.requires_grad = False

        #Input: 1@(30x30x30) 
        self.pretrain = nn.Sequential(*list(model.children()))
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(4096, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 1)
        
        self.drop1 = nn.Dropout(p=0.3)

    def forward(self, x1, x2):

        batch_sz = x1.size()[0]

        out1 = self.pretrain(x1)
        out1 = F.relu(self.fc1(out1))
       
        out2 = self.pretrain(x2)
        out2 = F.relu(self.fc1(out2))

        out = torch.cat((out1, out2), dim=1)
        
        out = F.relu(self.fc2(out))
        out = self.drop1(out)
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))

        return out


if __name__ == "__main__":

    # test out dimenssion
    A = Variable(torch.rand(32,1,30,30,30))
    print(A.size())
    A = A.cuda() 

    net = voxelNet(1)
    # net = voxelNet_fixed(1)

    for name, module in net.named_children():
        for n, parameters in module.named_parameters():
            print(n, parameters.requires_grad)
    print(net)
    # net = net.cuda()
    # out = net(A, A)
    # print(out.size())





