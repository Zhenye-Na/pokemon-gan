from utils.io_tools import read_dataset

# data = train_set = read_dataset("data/train.txt", "data/image_data/")

# print(len(data))
from PIL import Image
from skimage import io
from skimage.transform import resize
import numpy as np
import sklearn
import os

# imgdir = "./data/image_data"
# dstdir = "./data/resized_data"

# # If dest directory not exists, create dir
# if not os.path.isdir(dstdir):
#     os.mkdir(dstdir)

# for imname in os.listdir(imgdir):
#     img = io.imread(os.path.join(imgdir, imname))
#     img = resize(img, (256, 256), mode='reflect')
#     io.imsave(os.path.join(dstdir, imname), img)


# inpdir = dstdir
# outdir = "./data/preprocessed_data"

# # If output directory not exists, create dir
# if not os.path.isdir(outdir):
#     os.mkdir(outdir)

# process_method = 'rgb'

# if process_method == 'default':

#     # If the image channel is RGBA, then convert to
#     # gray-scale and back to RGB
#     for imname in os.listdir(inpdir):

#         # Image.open -> opens and identifies the given image file.
#         # Return -> an `Image` object.
#         img = Image.open(os.path.join(inpdir, imname))

#         # Check the original mode (channel for each image).
#         if img.mode == 'RGBA':
#             img.load() # required for img.split()

#             # Creates a new image using `L` mode -> gray-scale.
#             rgba2gray = Image.new("L", img.size)
#             # Convert from gray-scale to RGB.
#             gray2rgb = rgba2gray.convert(mode='RGB', colors=256)

#             gray2rgb.paste(img, mask=img.split()[3]) # 3rd is the alpha channel
#             gray2rgb.save(os.path.join(outdir, imname.split('.')[0] + '.jpg'), 'JPEG')
#         else:
#             img.convert(mode='RGB')
#             img.save(os.path.join(outdir, imname.split('.')[0] + '.jpg'), 'JPEG')

read_dataset("./data/image_data", "./data/preprocessed_data", 'default')
