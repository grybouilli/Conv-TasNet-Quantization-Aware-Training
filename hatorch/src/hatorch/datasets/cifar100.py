from torchvision import transforms, datasets
from .dataloader import DataLoader


class CIFAR100(DataLoader):
    """
    CIFAR100 DataLoader

    A DataLoader subclass for the CIFAR-100 dataset, which consists of 60000
    32x32 colour images in 100 classes, with 600 images per class.
    There are:
    - 500 training images per class
    - 100 testing images per class
    - 100 classes grouped into 20 superclasses.

    References
    ----------
    For more information, see:
    https://www.cs.toronto.edu/~kriz/cifar.html
    """
    image_size = (32, 32)

    def get_transforms(self):
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        train_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        valid_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            normalize,
        ])
        return train_transform, valid_transform

    def load_datasets(self, train_transform, valid_transform, download: bool = True):
        train_dataset = datasets.CIFAR100(
            root="./data", train=True, download=download, transform=train_transform
        )
        valid_dataset = datasets.CIFAR100(
            root="./data", train=False, download=download, transform=valid_transform
        )
        return train_dataset, valid_dataset
