from __future__ import print_function
import numpy as np
import os
import os.path
import sys
import random
import math
from pprint import pprint
from itertools import combinations
import torch
# from torch.autograd import Variable


"""
script to compute tripelts 
"""

def distance(vectors):
    return -2 * np.matmul(vectors, np.transpose(vectors)) + np.sum(np.square(vectors), axis=1).reshape((-1,1))+np.sum(np.square(vectors), axis=1).reshape((1,-1)) 

def compute_loss(loss_vectors, func):
    
    # loss_vectors = np.array(loss_vectors)
    if func == 'hard_negative':
        # returning the id which maximizes the loss
        idx = np.argmax(loss_vectors)
        if loss_vectors[idx] > 0:
            return idx
        else:
            return None
    
    elif func == 'random_hard':
        # returning the id of randomly selecting loss >0
        hard_negatives = np.where(loss_vectors > 0)[0]
        if len(hard_negatives) > 0:
            return np.random.choice(hard_negatives)
        else:
            return None

class TripletSampler():
    """
    Triplet sampler 
    """

    def __init__(self):
        pass

    def get_all_triplets(self, embeddings, labels):
        # get all triplets 

        # embeddings = embeddings.data.cpu().numpy()
        labels = labels.data.cpu().numpy()

        # print(set(labels))
        triplets = []
        if(len(set(labels))==1):
            # this means all labels are same (zero error)
            print('All labels same in triplet[%d]' %(len(labels)))
            triplets.append([0, 0, 0])
            return torch.cuda.LongTensor(np.array(triplets))
        
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2)) # All anchor-positive pairs
            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for 
                    anchor_positive in anchor_positives for neg_ind in negative_indices]

            triplets += temp_triplets

        # if(len(triplets)==0 and len(set(labels)) > 1):
        #     triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])
        # if(len(triplets)==0):
        #     # this means all labels are same (zero error)
            # triplets.append([0,0,0])
        
        return torch.cuda.LongTensor(np.array(triplets))
        # triplet_embeddings = get_embeddings(embeddings, triplets)
        # return triplet_embeddings 


    def get_random_triplets(self, embeddings, labels, size=32):
        # get random triplets

        can_triplets = self.get_all_triplets(embeddings, labels)

        if len(can_triplets) < size:
            random.shuffle(can_triplets)
            return can_triplets
        else:
            random.shuffle(can_triplets)
            return can_triplets[:size] 


    def get_negative_mining(self, embeddings, labels, margin, func='hard_negative',size=32):
        """perform neg mining all the possible triplets which high loss.
           func = hard_negative, random_hard 
        """
        embeddings = embeddings.data.cpu().numpy()
        labels = labels.data.cpu().numpy()

        similarity_matrix = distance(embeddings)
        # print(labels, type(labels))
        
        # print(set(labels))    
        triplets = []
        if(len(set(labels))==1):
            # this means all labels are same (zero error)
            print('All labels same in triplet[%d]' %(len(labels)))
            triplets.append([0, 0, 0])
            return torch.cuda.LongTensor(np.array(triplets))

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if(len(label_indices) < 2):
                continue

            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives= list(combinations(label_indices, 2)) 
            anchor_positives = np.array(anchor_positives)

            ap_distances = similarity_matrix[anchor_positives[:,0], anchor_positives[:,1]]

            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss = ap_distance - similarity_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + margin

                hard_negative = compute_loss(loss, func) 
                if hard_negative is not None:
                    triplets.append([anchor_positive[0], anchor_positive[1],
                                     negative_indices[hard_negative]])

        if(len(triplets)==0):
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])
                
        return torch.cuda.LongTensor(np.array(triplets))


if __name__ == "__main__":
    
    sampler = TripletSampler()

    # label = [1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8]
    # label = [1,2,3,1,2,3,1,2,3]
    label = [3,3,3,3,2,2]
    label = np.array(label)
    embed = np.random.rand(6, 5)

    embed = Variable(torch.from_numpy(embed))
    label = Variable(torch.from_numpy(label))
    
    embed = embed.cuda()
    label = label.cuda()

    # print(label.shape)
    # a = sampler.get_negative_mining(embed, label, margin=0.2)
    a = sampler.get_all_triplets(embed, label) 
    print(a, type(a))

    a_embed = torch.index_select(embed, 0, Variable(a[:,0]))
    b_embed = torch.index_select(embed, 0, Variable(a[:,1]))
    c_embed = torch.index_select(embed, 0, Variable(a[:,2]))

    # print(a_embed, b_embed, c_embed)


    # a = sampler.get_negative_mining(embed, label, margin=0.2, func='random_hard')
    # print(len(a), a, type(a))

