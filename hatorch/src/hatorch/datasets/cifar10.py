from torchvision import transforms, datasets
from .dataloader import DataLoader


class CIFAR10(DataLoader):
    """
    CIFAR10 DataLoader

    A DataLoader subclass for the CIFAR-10 dataset, which consists of 60000
    32x32 colour images in 10 classes, with 6000 images per class.
    There are:
    - 50000 training images
    - 10000 test images.
    - 10 classes (airplane, automobile, bird, cat, deer,
      dog, frog, horse, ship, truck).

    References
    ----------
    For more information, see:
    https://www.cs.toronto.edu/~kriz/cifar.html
    """
    image_size = (32, 32)

    def get_transforms(self):
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        train_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        valid_transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
        return train_transform, valid_transform

    def load_datasets(self, train_transform, valid_transform, download: bool = True):
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=download, transform=train_transform
        )
        valid_dataset = datasets.CIFAR10(
            root="./data", train=False, download=download, transform=valid_transform
        )
        return train_dataset, valid_dataset
