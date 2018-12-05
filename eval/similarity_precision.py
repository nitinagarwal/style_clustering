from __future__ import print_function
import numpy as np
import os
import os.path
import sys
import argparse
import random
import math
import time, datetime
from pprint import pprint

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data 

sys.path.append('../networks')
from voxelNet import voxelNet_fixed

sys.path.append('../utils')
from data_prep_util import *
from triplet_sampler import TripletSampler

from test_triplet_loader import getDataset

parser = argparse.ArgumentParser()

"""
Compute the similarity precision (accuracy) on \emph(ALL) the testing triplets
"""

parser.add_argument('--dataDir', type=str, default='', help='path to data Dir')
parser.add_argument('--data_type', type=str, default = 'mesh',  help='mesh/voxel/image')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, default=4, help='# of data loading workers')
parser.add_argument('--cuda', type=bool, default = True,  help='training with cuda')

parser.add_argument('--model_path', type=str, default = 'None',  help='pretrained model path')
parser.add_argument('--nweight', type=int, default = 8, help='# of weight matrices (M)')
parser.add_argument('--trans_inv', type=bool, default = True,  help='Translation Invariant')

opt = parser.parse_args()
print(opt)


# Define the network
if opt.data_type == 'mesh':
    pass
elif opt.data_type == 'voxel':
    # net = voxelNet(3dnin_path='../model/3dnin_fc_processed.pth')
    net = voxelNet_fixed(3dnin_path='../model/3dnin_fc_processed.pth')
elif opt.data_type == 'image':
    pass
else:
    raise NotImplementedError('choose a valid representation')


network_path = opt.model_path
net.load_state_dict(torch.load(network_path))


if opt.cuda:
    net.cuda()
    print('cuda testing')


testdataset = getDataset(data_dir=opt.dataDir,train=False, repres=opt.data_type)

testdataloader = torch.utils.data.DataLoader(testdataset, batch_size = opt.batchSize,
                                             shuffle=False, num_workers=opt.workers)

triplet_sampler = TripletSampler()
print('Total test models ', len(testdataset))


def compute_embedding():
    """
        computes the entire embedding
    """
    embed_list = torch.rand(1, 256).cuda()
    label_list = torch.rand(1,1).type(torch.cuda.LongTensor)

    if opt.data_type == 'voxel':
        
        for i, data in enumerate(testdataloader, 0):

            object_voxel, object_label = data
            object_voxel = Variable(object_voxel, volatile=True)
            object_label = Variable(object_label, volatile=True)

            if opt.cuda:
                object_voxel = object_voxel.cuda()
                object_label = object_label.cuda()
            
            object_embed = net(object_voxel)
        
            embed_list = torch.cat((embed_list, object_embed.data), 0)
            label_list = torch.cat((label_list, object_label.data), 0)
            
            del object_embed, object_voxel, object_label
            torch.cuda.empty_cache()

    elif opt.data_type == 'mesh':
        pass


    embed_list = embed_list[1:]
    label_list = label_list[1:]
    label_list = torch.squeeze(label_list)
    
    return embed_list, label_list


def main():
    net.eval()

    accuracy = 0

    object_embed, object_label = compute_embedding()
    object_embed, object_label = Variable(object_embed), Variable(object_label)
    # print(object_embed.size(), object_label.size())
    
    triplets = triplet_sampler.get_all_triplets(object_embed, object_label)
    # print(triplets.size())

    anchor_embed = torch.index_select(object_embed, 0, Variable(triplets[:,0]))
    positive_embed = torch.index_select(object_embed, 0, Variable(triplets[:,1]))
    negative_embed = torch.index_select(object_embed, 0, Variable(triplets[:,2]))
    # print(anchor_embed.size(), pos_embed.size(), neg_embed.size(), type(neg_embed))

    pos_distance = np.linalg.norm(anchor_embed.data - positive_embed.data, axis=1)
    neg_distance = np.linalg.norm(anchor_embed.data - negative_embed.data, axis=1)
   
    score = pos_distance - neg_distance
    accuracy += sum(1 for x in score if x < 0)


    total = float(len(triplets))
    print('Test triplets:', len(triplets)) 
    print('Total Accuracy on Test set is: %f' %(accuracy / total)) 


if __name__ == "__main__":
    start_time = time.time()
    main()
    time_exec = round(time.time() - start_time)
    print('Total time taken :', str(datetime.timedelta(seconds=time_exec)))
    


