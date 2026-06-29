from torchvision import transforms, datasets
from .dataloader import DataLoader


class MNIST(DataLoader):
    """
    MNIST DataLoader

    A DataLoader subclass for the MNIST dataset, which consists of 70,000
    grayscale images of handwritten digits (10 classes: 0–9).
    """
    image_size = (28, 28)

    def get_transforms(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        return transform, transform

    def load_datasets(self, train_transform, valid_transform, download: bool = True):
        train_dataset = datasets.MNIST(
            root="./data", train=True, download=download, transform=train_transform
        )
        valid_dataset = datasets.MNIST(
            root="./data", train=False, download=download, transform=valid_transform
        )
        return train_dataset, valid_dataset
