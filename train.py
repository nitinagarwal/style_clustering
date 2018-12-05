from __future__ import print_function
import numpy as np
import os
import os.path
import sys
import argparse
import random
import json
import math
import time, datetime
import visdom
from pprint import pprint
from termcolor import colored

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data 
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.init import xavier_normal, xavier_uniform
# from torchviz import make_dot

sys.path.append('./networks/')
from voxelNet import voxelNet_fixed
from triplet import TripletLoss_reverse 

sys.path.append('./utils/')
from data_prep_util import *
from provider import getDataset
from triplet_sampler import TripletSampler

parser = argparse.ArgumentParser()

# Dataset 
parser.add_argument('--dataDir', type=str, default='', help='path to data Dir')
parser.add_argument('--data_type', type=str, default = 'voxel',  help='mesh/voxel/image')
# parser.add_argument('--nvertices', type=int, default=10000, help='input vertices of the mesh')
# parser.add_argument('--nweight', type=int, default = 8, help='# of weight matrices (M)')
# parser.add_argument('--trans_inv', type=bool, default = True,  help='Translation Invariant')
parser.add_argument('--vox_sz', type=int, default = 30,  help='voxel size')
# parser.add_argument('--augment', type=bool, default = False,  help='Data Augmentation')

# Training 
# parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
# parser.add_argument('--MeshBatchSZ', type=int, default=8, help='Mesh batch size')
parser.add_argument('--workers', type=int, default=4, help='# of data loading workers')
parser.add_argument('--nepoch', type=int, default=500, help='# of training epochs')
parser.add_argument('--model', type=str, default = 'None',  help='pretrained model path')
parser.add_argument('--cuda', type=bool, default = False,  help='training with cuda')
parser.add_argument('--logf', type=str, default = 'log1',  help='log folder')
parser.add_argument('--save_nth_epoch', type=int, default = 2,  help='save network every nth epoch')
parser.add_argument('--rand_nth_epoch', type=int, default = 30,  help='triplet selection strategy-random triplets till nth epoch')

# Optimization 
parser.add_argument('--margin', type=float, default=0.001, help='triplet margin')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--wd', type=float, default=0, help='weight decay')
parser.add_argument('--momentum', type=float, default=0, help='momentum')
parser.add_argument('--lr_decay', type=float, default=0.1, help='Multiplicative factor used on learning rate at lr_steps')
parser.add_argument('--lr_steps', default=200, nargs="+", type=int ,help='List of epochs where the learning rate is decreased by lr_decay')

opt = parser.parse_args()
print(opt)

# creating directories
log_dir = os.path.join('./logs', opt.logf)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

#initialize learning curve on visdom, and color for input and output in visdom display
viz = visdom.Visdom(port = 8095)

graph_title = '%s Training[%s], lr=%0.7f; wd=%0.4f; margin=%0.3f; HNM=%d' %(opt.data_type, opt.logf, opt.lr, opt.wd, opt.margin, opt.rand_nth_epoch )

epoch_curve = viz.line(
    X = np.column_stack((np.array( [0] ), np.array([0]))),
    Y = np.column_stack((np.array( [0] ), np.array([0]))),
    opts=dict(title= graph_title, legend=['Train', 'Test'] ) 
)


# Loading Training and Testing Data
traindataset = getDataset(data_dir=opt.dataDir, train=True, repres=opt.data_type)

traindataloader = torch.utils.data.DataLoader(traindataset, batch_size = opt.batchSize, 
                                              shuffle=True, num_workers=opt.workers)

testdataset = getDataset(data_dir=opt.dataDir,train=False, repres=opt.data_type)

testdataloader = torch.utils.data.DataLoader(testdataset, batch_size = opt.batchSize,
                                             shuffle=False, num_workers=opt.workers)

print('Train Dataset:', len(traindataset))
print('Test Dataset:', len(testdataset)) 


# Initialize the loss functions
train_loss = AverageValueMeter()
test_loss = AverageValueMeter()


# Define the network
if opt.data_type == 'mesh':
    pass
elif opt.data_type == 'voxel':
    # net = voxelNet(3dnin_path='./model/3dnin_fc_processed.pth')
    net = voxelNet_fixed(3dnin_path='./model/3dnin_fc_processed.pth')
elif opt.data_type == 'image':
    pass
else:
    raise NotImplementedError('choose a valid representation')

    
# Initialize the parameters of the network using xavier initialization(only the last fc layers)
net = init_net(net, opt.cuda)

if opt.model != 'None': 
    network_path = opt.model
    net.load_state_dict(torch.load(network_path))
    net.cuda()
    print('loaded a trained model %s' %(opt.model))


# Setup the optimizer with its parameters
# optimizer = optim.Adam(net.parameters(), lr = opt.lr, weight_decay=opt.wd)
optimizer = optim.Adam(filter(lambda p : p.requires_grad, net.parameters()), lr = opt.lr, weight_decay=opt.wd)

# Setup the LR scheduler
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_steps, gamma=opt.lr_decay) 

# Setup the loss function
triplet_loss = TripletLoss_reverse(margin = opt.margin)

# Setup the triplet sampling strategy 
# triplet_sampler = TripletSampler()


# Logfile
logfile = os.path.join(log_dir, 'log.txt')
with open(logfile, 'a') as f: #open and append
        f.write('Train: ' + str(len(traindataset)) + ' Test: ' + str(len(testdataset)) +'\n')
        f.write(str(opt) + '\n')
        f.write(str(net) + '\n')

def train(ep):
    # training one epoch
    net.train()
    
    if opt.data_type == 'voxel':

        for i, data in enumerate(traindataloader, 0):

            optimizer.zero_grad()                   # zero all the gradients
            
            anchor, positive, negative = data
            anchor = Variable(anchor)
            positive = Variable(positive)
            negative = Variable(negative)

            if opt.cuda:
                anchor = anchor.cuda()
                positive = positive.cuda()
                negative = negative.cuda()

            d_ap = net(anchor, positive)
            d_an = net(anchor, negative)

            # Triplet Loss  
            loss_net = triplet_loss(d_ap, d_an)

            train_loss.update(loss_net.data[0]) 

            loss_net.backward()     # gradient computation
            optimizer.step()

            print('[%d: %d/%d] train loss: %f' %(ep, i, len(traindataloader),
                                                 loss_net.data[0]))
            
            del d_ap, d_an, anchor, positive, negative, loss_net
            torch.cuda.empty_cache()

    print(colored('%d: Avg. train loss: %f' %(ep, train_loss.avg), 'cyan')) 


def test(ep):
    # Testing on all test dataset triplets
    net.eval()
    
    if opt.data_type == 'voxel':
    
        for i, data in enumerate(testdataloader, 0):

            anchor, positive, negative = data
            anchor = Variable(anchor)
            positive = Variable(positive)
            negative = Variable(negative)

            if opt.cuda:
                anchor = anchor.cuda()
                positive = positive.cuda()
                negative = negative.cuda()

            d_ap = net(anchor, positive)
            d_an = net(anchor, negative)

            # Triplet Loss  
            loss_net = triplet_loss(d_ap, d_an)

            test_loss.update(loss_net.data[0]) 

            del d_ap, d_an, anchor, positive, negative, loss_net
            torch.cuda.empty_cache()

    print(colored('%d: Avg. test loss: %f' %(ep, test_loss.avg), 'cyan')) 
    
            

def main():

    best_test_loss = 1000000
    current_lr = opt.lr

    for epoch in range(opt.nepoch):
        train_loss.reset()
        test_loss.reset()

	scheduler.step()
        train(ep = epoch)
        test(ep = epoch)

        for param_group in optimizer.param_groups:
            if(current_lr != param_group['lr']):
                print('Learning rate has changed: %f to %f' %(current_lr, param_group['lr']))
            current_lr = param_group['lr']

        # Update the visom curves for both train and test
        viz.line(X = np.array([epoch]), 
            Y = np.array([train_loss.avg]), 
            win = epoch_curve,
            update = 'append', 
            name = 'Train',
            opts=dict(showlegend=True)
        )

        viz.line(X = np.array([epoch]), 
            Y = np.array([test_loss.avg]), 
            win = epoch_curve,
            update = 'append', 
            name='Test',
            opts=dict(showlegend=True)
        )

        # update best test_loss and save the net
        if best_test_loss > test_loss.avg:
            best_test_loss = test_loss.avg
            print('New best loss: ', best_test_loss)
            print('hence saving net ...')
            torch.save(net.state_dict(), os.path.join(log_dir,'best_net_'+str(epoch)+'.pth'))

        if (epoch+1) % opt.save_nth_epoch == 0:
            print('saving net ...')
            torch.save(net.state_dict(), os.path.join(log_dir,'siamese_net_'+str(epoch)+'.pth'))

	# Dump logs
	log_table = {
	    "train_loss" : train_loss.avg,
	    "test_loss" : test_loss.avg,
	    "epoch" : epoch,
	    "lr" : current_lr,
	    "besttest" : best_test_loss,
	}

	# print logtable
	with open(logfile, 'a') as f: #open and append
	    f.write('stats: ' + json.dumps(log_table) + '\n')


if __name__ == "__main__":
    start_time = time.time()
    main()
    time_exec = round(time.time() - start_time)
    print('Total time taken: ', str(datetime.timedelta(seconds=time_exec)))
    print('-------Done-----------')


