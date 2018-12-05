from __future__ import print_function
import numpy as np
import os
import os.path
import sys
import argparse
import math
from pprint import pprint

import torch
from torch.autograd import Variable
import torch.utils.data 

sys.path.append('../networks')
sys.path.append('../utils')
from data_prep_util import *
import binvox_rw as binvox
from provider import getDataset
from voxelNet import voxelNet
from graphNet import graphNet

"""
script to compute the pairwise distance between the dataset
"""

parser = argparse.ArgumentParser()

parser.add_argument('--query', type=int, nargs='+', default=[0], help='query index')
parser.add_argument('--K', type=int, default=10, help='top K ranks')
parser.add_argument('--model', type=str, default='None', help='trained model path')
parser.add_argument('--data_type', type=str, default='voxel', help='voxel,mesh,point')

opt = parser.parse_args()

# text file containing all the 537 meshes (both train and test data)
basepath = '/home/minions/Desktop/style_dataset/style_8k/test' # base folder 
# basepath = '/home/minions/Desktop/style_dataset/style_8k/train' # base folder 
trained_network_path = opt.model

cls = 'object' # style or object
data_type = opt.data_type # mesh, voxel, point, image
query = opt.query  # or the ids of meshes
K = opt.K # top K ranks of meshes similar to query

# graphNet parameters
nweight = 4
trans_inv = True
cuda = True

model_list = []
with open(os.path.join(basepath, 'test.txt')) as f:
# with open(os.path.join(basepath, 'train.txt')) as f:
    for line in f:
        line = line.strip()
        model_list.append(line)

# Initialize the network
if data_type == 'mesh':
    net = graphNet(nweight, trans_inv)
elif data_type == 'voxel':
    net = voxelNet()
else:
    pass

network_path = trained_network_path
net.load_state_dict(torch.load(network_path))
net.eval()

# groups items by object class or style
# nested dictionaries
if cls == 'object':
    # cls = ['bed', 'cabinet', 'chair', 'stool', 'table']
    score = {'bed':{}, 'cabinet':{}, 'chair':{}, 'stool':{}, 'table':{}}
    # for idx in range(cls):
    #     scores[cls[idx]]=[]
else:
    # cls = ['children', 'european', 'japanese', 'ming']
    score = {'children':{}, 'european':{}, 'japanese':{}, 'ming':{}}
    # for idx in range(cls):
    #     scores[cls[idx]]=[]
    
def compute_distance(basepath, model1_path, model2_path, data_type, net):

    model1_path = os.path.join(basepath, data_type, model1_path)
    model2_path = os.path.join(basepath, data_type, model2_path)

    if data_type == 'mesh':
        
        model1_ver, model1_face = load_obj_data(model1_path)
        model2_ver, model2_face = load_obj_data(model2_path)
        model1_adj, model1_adj_size = get_adjacency_matrix(model1_ver, model1_face)
        model2_adj, model2_adj_size = get_adjacency_matrix(model2_ver, model2_face)

        model1_ver = torch.from_numpy(model1_ver.astype(np.float32))
        model1_adj = torch.from_numpy(model1_adj.astype(np.float32))
        model1_adj_size = torch.from_numpy(model1_adj_size.astype(np.float32))

        model2_ver = torch.from_numpy(model2_ver.astype(np.float32))
        model2_adj = torch.from_numpy(model2_adj.astype(np.float32))
        model2_adj_size = torch.from_numpy(model2_adj_size.astype(np.float32))


        model1_ver, model1_adj, model1_adj_size = Variable(model1_ver), Variable(model1_adj), Variable(model1_adj_size) 
        model2_ver, model2_adj, model2_adj_size = Variable(model2_ver), Variable(model2_adj), Variable(model2_adj_size) 
        
        model1_ver = model1_ver.transpose(2,1)
        model2_ver = model2_ver.transpose(2,1)
       
        if cuda:
            net = net.cuda()
            model1_ver, model1_adj, model1_adj_size = model1_ver.cuda(), model1_adj.cuda(), model1_adj_size.cuda()
            model2_ver, model2_adj, model2_adj_size = model2_ver.cuda(), model2_adj.cuda(), model2_adj_size.cuda()
        
        model1_embed = net(model1_ver, model1_adj, model1_adj_size)
        model2_embed = net(model2_ver, model2_adj, model2_adj_size)
    
    elif data_type == 'voxel':

        path1 = os.path.splitext(model1_path)[0] + '.binvox'
        path2 = os.path.splitext(model2_path)[0] + '.binvox'

        with open(path1, 'rb') as f:
            model1 = binvox.read_as_3d_array(f)
            model1 = model1.data.astype(np.float32)
            model1 = torch.unsqueeze(torch.from_numpy(model1), 0)
            model1 = torch.unsqueeze(model1, 0)
        
        with open(path2, 'rb') as f:
            model2 = binvox.read_as_3d_array(f)
            model2 = model2.data.astype(np.float32)
            model2 = torch.unsqueeze(torch.from_numpy(model2), 0)
            model2 = torch.unsqueeze(model2, 0)

        model1, model2= Variable(model1), Variable(model2)

        if cuda:
            net = net.cuda()
            model1, model2= model1.cuda(), model2.cuda()

        model1_embed = net(model1)
        model2_embed = net(model2)

    elif data_type == 'image':
        pass
    elif data_type == 'point':
        pass

    model1_embed = torch.squeeze(model1_embed.data)
    model2_embed = torch.squeeze(model2_embed.data)

    # distance = model1_embed - model2_embed

    # distance = torch.mul(distance, distance) 
    # distance = torch.sum(distance)
    distance = (model1_embed - model2_embed).pow(2).sum()
    # distance = np.linalg.norm(model1_embed - model2_embed)
    return distance


# computing pairwise distance for all the models
if query == 'full':
    query = model_list
else:
    query = [model_list[x] for x in query]


matrix_score = {}
for idx1 in range(len(query)):

    query_path = query[idx1]
    # score = {}
    for idx in range(len(model_list)):

        distance = compute_distance(basepath, query_path, model_list[idx], data_type, net)
        
        name = model_list[idx].split('_')
       
        if cls == 'object':
            score[name[1].lower()][model_list[idx]] = distance
        else:
            score[name[0].lower()][model_list[idx]] = distance

            # score[model_list[idx]] = distance
            # item = {model_list[idx]:distance}
        
    matrix_score[query_path] = score
    # print(score)

    for k, v in matrix_score.items():

        print('Query mesh is: ', k)

        for k1, v1 in v.items():

            print('Object/Style is: ', k1)
            
            sorted_dic = sorted(v1.items(), key=lambda x: x[1]) 
            if(len(sorted_dic) < K):
                print('Top %d/%d results'%(len(sorted_dic), len(sorted_dic)))
                for idx in range(len(sorted_dic)):
                    print(sorted_dic[idx][0], ' ', sorted_dic[idx][1])
            else:
                print('Top %d/%d results'%(K, len(sorted_dic)))
                for idx in range(K):
                    print(sorted_dic[idx][0], ' ', sorted_dic[idx][1])



