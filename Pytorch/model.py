from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()


    def forward(self, input):

