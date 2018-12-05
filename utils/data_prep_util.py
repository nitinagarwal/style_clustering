from __future__ import print_function
import numpy as np
import sys
import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from plyfile import (PlyData, PlyElement)

# --------------------------------
# MESH IO
# --------------------------------

def load_ply_data(filename):
    """ read ply file, only vertices and faces """

    plydata = PlyData.read(filename)

    vertices = plydata['vertex'].data[:]
    vertices = np.array([[x, y, z] for x,y,z in vertices])

    # input are all traingle meshes
    faces = plydata['face'].data['vertex_indices'][:]
    faces = np.array([[f1, f2, f3] for f1,f2,f3 in faces])

    return vertices, faces

def save_ply_data(filename, vertex, face):
    """ save ply file, only vertices and faces """

    vertices = np.zeros(vertex.shape[0], dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    for i in range(vertex.shape[0]):
            vertices[i] = (vertex[i][0], vertex[i][1], vertex[i][2])
    # print(vertex, vertex.dtype)
   
    faces = np.zeros(face.shape[0], dtype=[('vertex_indices', 'i4', (3,))])
    for i in range(face.shape[0]):
            faces[i] = ([face[i][0], face[i][1], face[i][2]])
    # print(faces.shape, faces.dtype)

    e1 = PlyElement.describe(vertices, 'vertex')
    e2 = PlyElement.describe(faces, 'face')
    
    PlyData([e1, e2], text=True).write(filename)
    print('file saved')

def load_obj_data(filename):
    """
    A simply obj reader which reads vertices and faces only. 
    i.e. lines starting with v and f only
    """
    mesh = {}
    ver =[]
    fac = []
    if not path.endswith('obj'):
        sys.exit('the input file is not a obj file')

    with open(filename) as f:
        for line in f:
            if line.strip():
                inp = line.split()
                if(inp[0]=='v'):
                    ver.append([float(inp[1]), float(inp[2]), float(inp[3])])
                elif(inp[0]=='f'):
                    fac.append([float(inp[1]), float(inp[2]), float(inp[3])])

    V = np.array(ver)
    F = np.array(fac)
    
    return V, F


# --------------------------------
# Mesh Utils 
# --------------------------------

def jitter_vertices(vertices, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original shape 
        Output:
          Nx3 array, jittered shape 
    """
    N, C = vertices.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += vertices
    return jittered_data 

def rotate_vertices(vertices):
    """ Randomly rotate the points to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, input shape
        Output:
          Nx3 array, rotated shape
    """
    rotated_data = np.zeros(vertices.shape, dtype=np.float32)

    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])

    rotated_data = np.dot(vertices, rotation_matrix)
    return rotated_data

def rotate_vertices_by_angle(vertices, rotation_angle):
    """ Randomly rotate the points by rotation_angle to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, input shape
          rotation_angle in radians
        Output:
          Nx3 array, rotated shape
    """
    rotated_data = np.zeros(vertices.shape, dtype=np.float32)

    # rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])

    rotated_data = np.dot(vertices, rotation_matrix)
    return rotated_data


def normalize_shape(vertices):
    # normalize shape to fit inside a unit sphere
    ver_max = np.max(vertices, axis=0)
    ver_min = np.min(vertices, axis=0)
    
    centroid = np.stack((ver_max, ver_min), 0)
    centroid = np.mean(centroid, axis=0)
    vertices = vertices - centroid

    longest_distance = np.max(np.sqrt(np.sum((vertices**2), axis=1)))
    vertices = vertices / longest_distance

    return vertices


# --------------------------------
# Training Utils 
# --------------------------------

class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def model_summary(model, print_layers=False):
    train_count = 0
    nontrain_count = 0
    
    for name, p in model.named_parameters():
        if(p.requires_grad):
            if(print_layers):                 
                print('Train: ', name, 'has', p.numel())
            train_count += p.numel()
        
        elif not p.requires_grad:
            if(print_layers):
                print('Non Train: ', name, 'has', p.numel())
            nontrain_count += p.numel()
        
    print('Total Parameters: ', train_count+nontrain_count)    
    print('Trainable Parameters: ',train_count)
    print('NonTrainable Parameters: ',nontrain_count)


def init_net(net, cuda=True):
    """
    Initialize the network with xavier initialization
    only for the last fc layers
    """
    def initialize_weights(m):
        if(isinstance(m, nn.Conv1d)):
            xavier_normal(m.weight.data)

        elif(isinstance(m, nn.Linear)):
            xavier_normal(m.weight.data)

    net.apply(initialize_weights)

    if cuda:
        net.cuda()
        print('network on cuda')

    return net



if __name__ == '__main__':

    # filename = '../../scripts/model/chair_aligned_simplified/chair_0001.ply'
    # filename = '../../scripts/model/chair_aligned_simplified/8k/chair_0012.ply'
    filename = '/Users/Nitin/Desktop/8k/chair_0011.ply'

    # chair_aligned/chair_0016.ply'
    V, F = load_ply_data(filename)
    print(V.shape, F.shape)
    # print(F)
    # wfile = '../../scripts/model/chair/test.ply'
    # save_ply_data(wfile, V, F)
    # F = F[0:10,:]
    # V = V[0:20,:]

    # adj, adj_size = get_adjacency_matrix(V, F)
    Q = compute_Q_matrix(V, F)
    # print(Q)
    print(np.shape(Q), type(Q))

    # print([adj[i,:] for i in range(len(adj_size))])
    # print(adj.shape, adj_size.shape)
    # print(adj, adj_size)



