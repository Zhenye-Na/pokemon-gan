# from utils.io_tools import read_dataset

# data = train_set = read_dataset("data/train.txt", "data/image_data/")

# print(len(data))

from skimage import io
from skimage.transform import resize
import numpy as np
import sklearn
import os

imgdir = "./data/image_data"
dstdir = "./data/resized_data"

os.mkdir(dstdir)

for imname in os.listdir(imgdir):
    img = io.imread(os.path.join(imgdir, imname))
    img = resize(img, (256, 256), mode='reflect')
    io.imsave(os.path.join(dstdir, imname), img)