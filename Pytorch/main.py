"""DCGAN implementation in Pytorch of Pokemon GAN.

High level pipeline

"""

import torch
import argparse
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms

from utils.Pokemon import Pokemon


# Parameters settings
parser = argparse.ArgumentParser(description="Pokemon GAN")

parser.add_argument('--dataroot', type=str, default="../data/preprocessed_data", help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--niters', type=int, default=200, help='number of epochs to train')
parser.add_argument('--resume', type=bool, default=False, help='resume training or not')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()

# ---------------------------- #
# Generate seeds
np.random.seed(111)
torch.cuda.manual_seed_all(111)
torch.manual_seed(111)
# ---------------------------- #


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = Pokemon(root=opt.dataroot, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=opt.workers)
print("Train set size: " + str(len(trainset)))
