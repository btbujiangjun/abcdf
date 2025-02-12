import math
from torchvision import utils

def save_images(images, file_name):
    utils.save_image(
        images,
        file_name,
        nrow = int(math.sqrt(images.shape[0]))
    )

