from __future__ import print_function
import numpy as np
import os
import os.path
import sys
import pymesh
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data
from pprint import pprint
from termcolor import colored

sys.path.append('../utils')
from data_prep_util import *
import binvox_rw as binvox


class getDataset(data.Dataset):

    """ Normal dataset loader for Evaluation
        Read the training/testing from file along with their labels     
    """
    
    def __init__(self, data_dir, train=True, repres='voxel'):

        self.root = data_dir
        self.representation = repres # voxel, mesh, image
        self.datapath = [] 

        if train:
            self.file = os.path.join(self.root, 'train.txt')
            self.label_file = os.path.join(self.root, 'train_label.txt')
            self.mode = 'train'

            with open(self.file) as f:
                for line in f:
                    line = line.split()
                    path = os.path.join(self.root, self.representation, line[0])
                    self.datapath.append(path)

            self.label = [] 
            with open(self.label_file) as f:
                for line in f:
                    line = line.split()
                    self.label.append(int(line[0]))

        else:
            self.file = os.path.join(self.root, 'test.txt')
            self.label_file = os.path.join(self.root, 'test_label.txt')
            self.mode = 'test'

            with open(self.file) as f:
                for line in f:
                    line = line.split()
                    path = os.path.join(self.root, self.representation, line[0])
                    self.datapath.append(path)

            self.label = [] 
            with open(self.label_file) as f:
                for line in f:
                    line = line.split()
                    self.label.append(int(line[0]))


    def __getitem__(self, index):

       model_path = self.datapath[index]
       model_label = np.array([self.label[index]])

       # read voxels or images or meshes 
       if self.representation=='voxel':
           model_path = os.path.splitext(model_path)[0] + '.binvox'

           with open(model_path, 'rb') as f:
               model = binvox.read_as_3d_array(f)
               model = model.data.astype(np.float32)
       
           # convert to torch 
           model = torch.unsqueeze(torch.from_numpy(model), 0)
       
       elif self.representation=='mesh':
           pass

       elif self.representation == 'image':
           pass

       return [model, model_label]


    def __len__(self):
        return len(self.datapath)

  
