import torch
import torch.nn as nn
import torch.legacy.nn as lnn

from functools import reduce
from torch.autograd import Variable

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))


voxel_nin = nn.Sequential( # Sequential,
	nn.Conv3d(1,48,(6, 6, 6),(2, 2, 2),(0, 0, 0)),
	nn.BatchNorm3d(48,1e-05,0.1,True),#BatchNorm3d,
	nn.ReLU(),
	nn.Conv3d(48,48,(1, 1, 1),(1, 1, 1),(0, 0, 0)),
	nn.BatchNorm3d(48,1e-05,0.1,True),#BatchNorm3d,
	nn.ReLU(),
	nn.Conv3d(48,48,(1, 1, 1),(1, 1, 1),(0, 0, 0)),
	nn.BatchNorm3d(48,1e-05,0.1,True),#BatchNorm3d,
	nn.ReLU(),
	nn.Dropout(0.2),
	nn.Conv3d(48,96,(5, 5, 5),(2, 2, 2),(0, 0, 0)),
	nn.BatchNorm3d(96,1e-05,0.1,True),#BatchNorm3d,
	nn.ReLU(),
	nn.Conv3d(96,96,(1, 1, 1),(1, 1, 1),(0, 0, 0)),
	nn.BatchNorm3d(96,1e-05,0.1,True),#BatchNorm3d,
	nn.ReLU(),
	nn.Conv3d(96,96,(1, 1, 1),(1, 1, 1),(0, 0, 0)),
	nn.BatchNorm3d(96,1e-05,0.1,True),#BatchNorm3d,
	nn.ReLU(),
	nn.Dropout(0.2),
	nn.Conv3d(96,512,(3, 3, 3),(2, 2, 2),(0, 0, 0)),
	nn.BatchNorm3d(512,1e-05,0.1,True),#BatchNorm3d,
	nn.ReLU(),
	nn.Conv3d(512,512,(1, 1, 1),(1, 1, 1),(0, 0, 0)),
	nn.BatchNorm3d(512,1e-05,0.1,True),#BatchNorm3d,
	nn.ReLU(),
	nn.Conv3d(512,512,(1, 1, 1),(1, 1, 1),(0, 0, 0)),
	nn.BatchNorm3d(512,1e-05,0.1,True),#BatchNorm3d,
	nn.ReLU(),
	nn.Dropout(0.2),
	Lambda(lambda x: x.view(x.size(0),-1)), # View,
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,512)), # Linear,
	nn.ReLU(),
	nn.Dropout(0.5),
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(512,40)), # Linear,
)