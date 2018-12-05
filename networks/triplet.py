import torch.nn as nn
import torch.nn.functional as F
import torch
# from torch.autograd import Variable

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    small distance_positive = anchor and positive are similar
    high distance_negative = anchor and negative are dissimilar
    """

    def __init__(self, margin=0.01):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)

        return losses.mean() if size_average else losses.sum()


class TripletLoss_reverse(nn.Module):
    """
    Triplet loss in reverse
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    high similarity_ap = anchor and positive are similar
    low similarity_an = anchor and negative are dissimilar
    """

    def __init__(self, margin=0.01):
        super(TripletLoss_reverse, self).__init__()
        self.margin = margin

    def forward(self, similarity_ap, similarity_an, size_average=True):
        
        losses = F.relu(similarity_an - similarity_ap + self.margin)

        return losses.mean() if size_average else losses.sum()


if __name__ == "__main__":
    # a = Variable(torch.rand(1,5)).cuda()
    a = Variable(torch.zeros(1,5)).cuda()
    b = a
    c = a
    print(a, b, c)

    loss= TripletLoss()

    ls = loss(a, b, c, False)

    print(ls.data[0])
