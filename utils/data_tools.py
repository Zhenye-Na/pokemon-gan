"""Image resize for Input of Pokemon GAN."""
import os
import skimage
from skimage import io
from skimage.transform import resize

src = "./data" #pokeRGB_black
dst = "./resizedData" # resized

os.mkdir(dst)

for each in os.listdir(src):
    img = io.imread(os.path.join(src, each))
    img = resize(img, (256, 256))
    io.imsave(os.path.join(dst, each), img)


# skimage.io.imsave(fname, arr, plugin=None, **plugin_args)[source]
# Save an image to file.

# Parameters:
# fname : str

# Target filename.

# arr : ndarray of shape (M,N) or (M,N,3) or (M,N,4)

# Image data.

# plugin : str

# Name of plugin to use. By default, the different plugins are tried (starting
# 	with the Python Imaging Library) until a suitable candidate is found. If
# not given and fname is a tiff file, the tifffile plugin will be used.
#