"""NumPy arrays (N, H, W, 1) -> PyTorch tensors (N, 1, H, W) and loaders."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def to_tensor(array):
    """(N, H, W, C) NumPy array -> (N, C, H, W) float32 tensor."""
    return torch.from_numpy(
        np.ascontiguousarray(np.asarray(array, dtype=np.float32)
                             .transpose(0, 3, 1, 2)))


def to_numpy(tensor):
    """(N, C, H, W) tensor -> (N, H, W, C) NumPy array."""
    return tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)


def make_loader(images, masks, batch_size, shuffle=False, seed=None):
    """Batches in the given order, or reshuffled every epoch.

    With shuffle=True the order of each epoch comes from a generator seeded
    with `seed`, so it differs between epochs but is the same in every run.
    """
    dataset = TensorDataset(to_tensor(images), to_tensor(masks))
    generator = None
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0)
