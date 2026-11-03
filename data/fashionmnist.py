import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_fashion_loaders(batch_size, num_workers=2):
    """Fashion-MNIST loaders with pixels in [0, 1] (no normalization)."""
    transform = transforms.ToTensor()

    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
