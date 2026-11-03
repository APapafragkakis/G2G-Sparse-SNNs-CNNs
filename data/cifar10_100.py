import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))

# Mean/std used to train the pretrained chenyaofo/pytorch-cifar-models ResNets.
# Only applied when normalize=True (the frozen ResNet feature extractor path).
CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN, CIFAR100_STD = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)


def _loaders(dataset_cls, mean, std, batch_size, normalize, num_workers):
    steps = [transforms.ToTensor()]
    if normalize:
        steps.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(steps)

    train_set = dataset_cls(DATA_ROOT, train=True, download=True, transform=transform)
    test_set = dataset_cls(DATA_ROOT, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def get_cifar10_loaders(batch_size, normalize=False, num_workers=2):
    """CIFAR-10 loaders. Pixels in [0, 1], or normalized for the pretrained ResNet."""
    return _loaders(datasets.CIFAR10, CIFAR10_MEAN, CIFAR10_STD, batch_size, normalize, num_workers)


def get_cifar100_loaders(batch_size, normalize=False, num_workers=2):
    """CIFAR-100 loaders. Pixels in [0, 1], or normalized for the pretrained ResNet."""
    return _loaders(datasets.CIFAR100, CIFAR100_MEAN, CIFAR100_STD, batch_size, normalize, num_workers)
