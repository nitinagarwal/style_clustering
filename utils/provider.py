from __future__ import print_function
import numpy as np
import os
import os.path
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data
from pprint import pprint

from data_prep_util import *
import binvox_rw as binvox

class getDataset(data.Dataset):

    """ Triplet dataset for training siamese networks
        Read the training/testing triplets from file     
    """
    
    def __init__(self, data_dir, train=True, repres='voxel'):

        self.root = data_dir
        self.representation = repres # voxel, mesh, image

        if train:
            self.file = os.path.join(self.root, 'train_triplets.txt')
            self.mode = 'train'
        else:
            self.file = os.path.join(self.root, 'test_triplets.txt')
            self.mode = 'test'

        self.triplets = [] 
        with open(self.file) as f:
            for line in f:
                line = line.split()
                anch = os.path.join(self.root, self.representation, line[0]) 
                pos = os.path.join(self.root, self.representation, line[1]) 
                neg = os.path.join(self.root, self.representation, line[2]) 

                self.triplets.append((anch, pos, neg))


    def __getitem__(self, index):

       anch, pos, neg = self.triplets[index]

       # read voxels or images or meshes 
       if self.representation=='voxel':
           anch = os.path.splitext(anch)[0] + '.binvox'
           pos = os.path.splitext(pos)[0] + '.binvox'
           neg = os.path.splitext(neg)[0] + '.binvox'

           with open(anch, 'rb') as f:
               anch_model = binvox.read_as_3d_array(f)
               anch_model = anch_model.data.astype(np.float32)
       
           with open(pos, 'rb') as f:
               pos_model = binvox.read_as_3d_array(f)
               pos_model = pos_model.data.astype(np.float32)
       
           with open(neg, 'rb') as f:
               neg_model = binvox.read_as_3d_array(f)
               neg_model = neg_model.data.astype(np.float32)
       
           # convert to torch 
           anch_model = torch.unsqueeze(torch.from_numpy(anch_model), 0)
           pos_model = torch.unsqueeze(torch.from_numpy(pos_model), 0)
           neg_model = torch.unsqueeze(torch.from_numpy(neg_model), 0)
       
       elif self.representation=='mesh':
           pass

       elif self.representation == 'image':
           pass

       return [ anch_model, pos_model, neg_model ]


    def __len__(self):
        return len(self.triplets)

  

if __name__ == "__main__":

    # dataset = getDataset(data_dir='../../style_data', train=True, repres='voxel')
    dataset = getDataset(data_dir='/home/minions/Desktop/style_dataset/style_8k', train=False, repres='voxel')

    print('lenth of dataset', len(dataset))

    # a = dataset[0]
    a, p, n = dataset[0]
    # print(a, n)
    print(a.shape, p.shape, n.shape)
    print(type(a))

