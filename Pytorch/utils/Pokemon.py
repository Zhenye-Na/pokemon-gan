"""Initialize Pokemon class for data preprocessing and augmentation."""


from __future__ import print_function
from PIL import Image
import os
import numpy as np
from skimage import io


class Pokemon(object):
    """`Pokemon <https://www.floydhub.com/zayne/datasets/preprocessedpokemon>` Dataset.

    Args:
        root (string): Root directory of dataset where directory.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """
    def __init__(self, root, transform=None):
        """Initialize Pokemon."""
        super(Pokemon, self).__init__()

        self.transform = transform
        self.root = root

        self.train_data = []
        # base_folder = '../../data/preprocessed_data'
        self.train_list = os.listdir(self.root)

        for file in self.train_list:
            imgdir = os.path.join(self.root, file)
            self.train_data.append(io.imread(imgdir))

        self.train_data = np.array(self.train_data)

    def __getitem__(self, index):
        """Get item from Pokemon class.

        Args:
            index (int): Index
        Returns:
            img: PIL Image.
        """
        img = self.train_data[index]

        # Return a PIL Image for later agmentation
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        """Get length."""
        return len(self.train_data)



