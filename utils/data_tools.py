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

        if process_method is 'default':
          1. Convert images to range [0,1].
          2. Convert from rgba to gray-scale then back to rgb. Using skimage.
          (if possible)

        if process_method is 'rgb'
          1. Convert the images to range of [0, 1].
          2. Convert from rgba to rgb. Using skimage. (if possible)

        if process_method is 'hsv':
          1. Convert images to range [0,1].
          2. Convert from rgba to hsv. Using skimage. (if possible)

    Returns:
        Nothing to return.
        Preprocessed image will be automatically saved in file
    """

    # First, we will resize all the images for pokemon to the same size,
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

    # Second, we preprocess all of the image based on
    # selected methods, it may give different accuracy.
    # Change input dir for all of the images.
    inpdir = dstdir
    outdir = "../data/preprocessed_data"

    # If output directory not exists, create dir
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    if process_method == 'default':

        # If the image channel is RGBA, then convert to
        # gray-scale and back to RGB
        for imname in os.listdir(inpdir):

            # Image.open -> opens and identifies the given image file.
            # Return -> an `Image` object.
            img = Image.open(os.path.join(inpdir, imname))

            # Check the original mode (channel for each image).
            if img.mode == 'RGBA':
                # required for img.split()
                img.load()

                # Creates a new image using `L` mode -> gray-scale.
                rgba2gray = Image.new("L", img.size)
                # Convert from gray-scale to RGB.
                gray2rgb = rgba2gray.convert(mode='RGB', colors=256)

                # 3rd is the alpha channel
                gray2rgb.paste(img, mask=img.split()[3])
                gray2rgb.save(os.path.join(outdir, imname.split('.')[0] + '.jpg'), 'JPEG')
            else:
                img.convert(mode='RGB')
                img.save(os.path.join(outdir, imname.split('.')[0] + '.jpg'), 'JPEG')

    elif process_method == 'rgb':
        # If the image channel is RGBA, then convert to RGB
        pass

    elif process_method == 'hsv':
        # If the image channel is RGBA, then convert to hsv
        pass

    else:
        print("Method" + process_method + "is supported here. You wanna give it a try on your own? :)")
