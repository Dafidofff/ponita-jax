import os
import requests
import zipfile
import pickle
import numpy as np
import jax.numpy as jnp

from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from tqdm import tqdm


class MNISTPointCloud(Dataset):

    def __init__(
            self,
            root='./datasets/mnist_point_cloud',
            split='train',
            download=True,
    ):
        self.mnist = MNIST(root=root, train=True if split == 'train' else False, download=download)

        self.mean = 4.4539332
        self.std = 2.8892703

        # Create grid of points for MNIST images
        self.pos = np.stack(np.meshgrid(np.linspace(-1, 1, 28), np.linspace(-1, 1, 28)), axis=-1).reshape(-1, 2)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        # Create a point cloud from the image
        image, label = self.mnist[idx]
        pos = self.pos
        image = np.array(image) / 255.0
        image = image.flatten()[..., np.newaxis]
        # normalize
        image = (image - self.mean) / self.std
        # Create a dictionary with the point cloud and the label
        return {'pos': pos, 'x': image, 'y': label, 'mask': np.ones_like(image)}


def collate_fn(batch):
    """Collate function for the MNIST point cloud dataset."""
    keys = ['pos', 'x', 'y', 'mask']

    batch_dict = {k: [d[k] for d in batch] for k in keys}
    for k in ['pos', 'x', 'y', 'mask']:
        batch_dict[k] = np.stack(batch_dict[k], axis=0)
    batch_dict['edge_index'] = batch_dict['mask']
    batch_dict['batch'] = np.arange(len(batch_dict['x']))
    return batch_dict


