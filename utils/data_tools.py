"""Image resize for Input of Pokemon GAN."""
import os
import skimage
from skimage import io
from skimage.transform import resize
from PIL import Image


def preprocess_data(data_txt_file, image_data_path, process_method='default'):
	"""Preprocesses dataset.

    Args:
        data(dict): Python dict loaded using io_tools.
        process_method(str): processing methods needs to support
          ['raw', 'default'].

        if process_method is 'raw'
          1. Convert the images to range of [0, 1].
          2. Convert from rgba to rgb. Using skimage. (if possible)

        if process_method is 'default':
          1. Convert images to range [0,1].
          2. Convert from rgba to gray-scale then back to rgb. Using skimage. (if possible)

        if process_method is 'custom':
          1. Convert images to range [0,1].
          2. Convert from rgba to hsv. Using skimage. (if possible)

    Returns:
        Nothing to return. Preprocessed image will be automatically saved in file
    """

    # First we will resize all the images for pokemon to the same size,
	# in order to feed into DCGAN.
    imgdir = "../data/image_data"
    dstdir = "../data/resized_data"

    # If dest directory not exists, create dir
    if not os.path.isdir(dstdir):
    	os.mkdir(dstdir)

    # Resize all the images to (256, 256)
    for imname in os.listdir(imgdir):
        img = io.imread(os.path.join(imgdir, imname))
        img = resize(img, (256, 256), mode='reflect')
        io.imsave(os.path.join(dstdir, imname), img)

    # Change dir for dataset
    inpdir = dstdir
    outdir = "../data/preprocessed_data"

    if process_method == 'default':
        # If the image channel is RGBA, then convert to 
        # gray-scale and back to RGB
        for imname in os.listdir(inpdir):
        img = Image.open(os.path.join(inpdir, imname))

        if img.mode == 'RGBA':
            img.load() # required for img.split()
            background = Image.new("RGB", img.size, (0,0,0))
            background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
            background.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')
        else:
            img.convert('RGB')
            img.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')




    elif process_method == 'raw':
    	pass



    elif process_method == 'custom':
    	pass



    else:
    	print("Method" + process_method + "is supported here, why don't you give a try on your own? :)")



imgsrc = "../data/resized-data"
dst = "../data/resized_black/"

for each in os.listdir(src):
    png = Image.open(os.path.join(src,each))
    # print each
    if png.mode == 'RGBA':
        png.load() # required for png.split()
        background = Image.new("RGB", png.size, (0,0,0))
        background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
        background.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')
    else:
        png.convert('RGB')
        png.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')




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