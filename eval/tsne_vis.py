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

from tsne import bh_sne
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

sys.path.append('../networks')
sys.path.append('../utils')
from data_prep_util import *
import binvox_rw as binvox
from provider import getDataset
from voxelNet import voxelNet
from graphNet import graphNet

"""
script to compute the tsne embedding and visualization
"""

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='None', help = 'trained model_path')
parser.add_argument('--data_type', type=str, default='voxel', help = 'voxel, mesh, point')

opt = parser.parse_args()

# text file containing all the 518 meshes (both train and test data)
# trained_network_path = '/home/minions/Dropbox/GraphicsLab/Projects/3D_Content_Creation/code/detail_estimator/logs/log4/siamese_network.pth'

trained_network_path = opt.model
basepath = '/home/minions/Desktop/style_dataset/style_8k/test'   # base folder
# cls = 'object' # style or object
data_type = opt.data_type # mesh, voxel, point, image
# query = 'full'  # or the ids of meshes

# voxelNet parameters
nweight = 4
trans_inv = True
cuda = True

model_list = []
with open(os.path.join(basepath, 'test.txt')) as f:
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

def compute_embedding(basepath, model1_path, data_type, net):

    model1_path = os.path.join(basepath, data_type, model1_path)

    if data_type == 'mesh':
        
        model1_ver, model1_face = load_obj_data(model1_path)
        model1_adj, model1_adj_size = get_adjacency_matrix(model1_ver, model1_face)

        model1_ver = torch.from_numpy(model1_ver.astype(np.float32))
        model1_adj = torch.from_numpy(model1_adj.astype(np.float32))
        model1_adj_size = torch.from_numpy(model1_adj_size.astype(np.float32))

        model1_ver, model1_adj, model1_adj_size = Variable(model1_ver), Variable(model1_adj), Variable(model1_adj_size) 
        
        model1_ver = model1_ver.transpose(2,1)
       
        if cuda:
            net = net.cuda()
            model1_ver, model1_adj, model1_adj_size = model1_ver.cuda(), model1_adj.cuda(), model1_adj_size.cuda()
        
        model1_embed = net(model1_ver, model1_adj, model1_adj_size)
    
    elif data_type == 'voxel':
        
        path1 = os.path.splitext(model1_path)[0] + '.binvox'

        with open(path1, 'rb') as f:
            model1 = binvox.read_as_3d_array(f)
            model1 = model1.data.astype(np.float32)
            model1 = torch.unsqueeze(torch.from_numpy(model1), 0)
            model1 = torch.unsqueeze(model1, 0)
        
        model1 = Variable(model1) 
        
        if cuda:
            net = net.cuda()
            model1 = model1.cuda()

        model1_embed = net(model1)

    elif data_type == 'image':
        pass
    elif data_type == 'point':
        pass
   
    model1_embed = torch.squeeze(model1_embed.data)
    model1_embed = model1_embed.cpu()
    
    return model1_embed.numpy()


def imscatter(x, y, path, ax=None, zoom=1):

    artists = []
    root='/home/minions/Desktop/euro_simplify_data/8k/imgs' # root folder for images
    
    
    for i in range(len(x)):

        image = plt.imread(os.path.join(root, os.path.splitext(path[i])[0] + '.png'))
        im = OffsetImage(image, zoom=zoom)
        ab = AnnotationBbox(im, [x[i], y[i]], xycoords='data', frameon=True)
        artists.append(ax.add_artist(ab))
   
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def evaluate_tsne(model_filenames, latent_embedding, perplexity=30):
    """analyzing tsne plot"""

    # print(len(latent_embedding), type(latent_embedding[0]))
    latent_embedding = np.asarray(latent_embedding).astype('float64')
    print(latent_embedding.shape)

    vis_data = bh_sne(latent_embedding, perplexity=perplexity)

    # plot the result
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]
    

    fig, ax =plt.subplots()
    imscatter(vis_x, vis_y, ax=ax, path=model_filenames, zoom=0.1)
    # plt.scatter(vis_x, vis_y, cmap=plt.cm.get_cmap("jet", 10))
    # plt.colorbar(ticks=range(10))
    # plt.clim(-0.5, 9.5)
    # plotly_fig = tls.mpl_to_plotly(fig)
    # viz._send({'data':plotly_fig.data, 'layout':plotly_fig.layout}) 
    # pickle.dump(fig,file('test.pickle','w')) 
    plt.show()


if __name__ == "__main__":
   
    vector = []
    # compute embedding
    for idx in range(len(model_list)):

        embedd = compute_embedding(basepath, model_list[idx], data_type, net)
        # print(type(embedd), embedd.shape)
        vector.append(embedd)

    evaluate_tsne(model_list, vector, perplexity = 10)




