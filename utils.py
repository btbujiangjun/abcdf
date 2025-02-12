"""
Save a batch of images as a single grid image.

Author: Jiang Jun
Date: 2025-02-11
"""

import math
from torchvision import utils

def save_images(images, file_name):
    utils.save_image(
        images,
        file_name,
        nrow = int(math.sqrt(images.shape[0]))
    )

