"""
This script contains custom Dataset and DataLoader classes for loading and transforming images.
   - ABCDataset: A custom dataset class that loads images from a specified folder and applies transformations.
   - ABCDataLoader: A custom DataLoader class that batches the data and optionally shuffles it.
   - Supports image loading with recursive file searching based on extensions.

Author: Jiang Jun
Date: 2025-02-11
"""

import os
from PIL import Image
from pathlib import Path
from multiprocessing import cpu_count
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

class ABCDataset(Dataset):
    """
    Custom Dataset to load and transform images from a given folder.

    Args:
        folder (str): Path to the folder containing images.
        image_size (int): Desired size for resizing images.
        file_ext (str): File extension or pattern to match image files (default: "*.jpg").
    """
    def __init__(self, 
            folder:str, 
            image_size:int=64,
            file_ext:str="**/*.jpg"
        ):
        assert os.path.exists(folder), f"Folder not found:{folder}."
        
        self.images = list(Path(folder).rglob(file_ext))
        
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])  

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Get an image by index and apply the necessary transformations.
        
        Args:
            index (int): Index of the image to fetch.
        
        Returns:
            torch.Tensor: Transformed image tensor.
        """
        try:
            image = Image.open(self.images[index])
            image = image.convert('RGB')  # Ensure 3-channel image
            return self.transform(image)
        except Exception as e:
            raise RuntimeError(f"Error loading image {self.images[index]}: {e}")

class ABCDataLoader(DataLoader):
    """
    Custom DataLoader for ABCDataset to load and batch the images.

    Args:
        dataset (ABCDataset): The dataset to load.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data (default: False).
        pin_memory (bool): Whether to pin memory for faster data transfer (default: True).
        num_workers (int): Number of workers for data loading (default: number of CPU cores).
        drop_last (bool): Whether to drop the last incomplete batch (default: False).
    """
    def __init__(self, 
            dataset:ABCDataset,
            batch_size:int,
            shuffle:bool = False,
            pin_memory:bool = True,
            num_workers:int = cpu_count(),
            drop_last: bool = False,
            *args,
            **kwargs,):
        super().__init__(dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            pin_memory=pin_memory, 
            num_workers=num_workers, 
            drop_last=drop_last,
            *args, 
            **kwargs
        )
        self.size = len(dataset)
        self.num_batch = len(dataset) // batch_size

