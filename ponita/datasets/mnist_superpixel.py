import numpy as np

from torch.utils.data import Dataset
from torch_geometric.datasets import MNISTSuperpixels


class MNISTSuperPixelPointCloud(Dataset):

    def __init__(
            self,
            root='./datasets/mnist_point_cloud',
            split='train',
    ):
        self.mnist = MNISTSuperpixels(root=root, train=True if split == 'train' else False, transform=None)

    def __len__(self):
        return len(self.mnist) 

    def __getitem__(self, idx):
        graph = self.mnist[idx]
        return graph


def collate_fn(batch):
    """Collate function for the MNIST point cloud dataset."""
    keys = ['pos', 'x', 'y']

    batch_dict = {k: [d[k] for d in batch] for k in keys}
    for k in ['pos', 'x', 'y']:
        batch_dict[k] = np.stack(batch_dict[k], axis=0)
    batch_dict['y'] = batch_dict['y'].squeeze()
    return batch_dict

ds = MNISTSuperPixelPointCloud()
from torch.utils.data import DataLoader
dl = DataLoader(ds, batch_size=5, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=True)

smp = next(iter(dl))