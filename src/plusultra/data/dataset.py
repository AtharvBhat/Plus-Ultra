"""
File that defines the pytorch dataset class to load data
"""

import pathlib

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import to_tensor

import plusultra.data.transforms as transforms
import plusultra.utils as utils


class UnsplashDataset(Dataset):
    """Dataset Class for unsplash Datset"""

    def __init__(
        self,
        path: str = "data/raw/unsplash",
        transforms: list[transforms.ImageTransform] = [transforms.IdentityTransform()],
        size: int = 256,
    ) -> None:
        """Initialise unsplash dataset

        Args:
            transforms (list[ImageTransform]):
                List of image transforms to be applied to the input
            path (str, optional): Path to raw unsplash dataset.
                Defaults to "data/raw/unsplash".
            size (int, optional): Size of Target images
                (Raw images will be randomly cropped to be of this size)
        """
        super().__init__()
        self.image_list = [
            path
            for path in pathlib.Path(f"{utils.get_project_root()}/{path}").rglob(
                "*.jpg"
            )
        ]
        self.transforms = transforms
        self.size = size

    def __len__(self) -> int:
        """Returns total number of samples in dataset

        Returns:
            int: number of images in unsplash dataset
        """
        return len(self.image_list)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns an input and a desired target for the model

        Args:
            index (int): index for in image list

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple[input, target]
        """
        image = Image.open(self.image_list[index])
        image = RandomCrop(self.size)(image)
        target = image.copy()

        for transform in self.transforms:
            image = transform(image)

        return to_tensor(image), to_tensor(target)


class Div2kDataset(UnsplashDataset):
    """Dataset Class for Div2k Datset"""

    def __init__(
        self,
        path: str = "data/raw/div2k/train",
        transforms: list[transforms.ImageTransform] = [transforms.IdentityTransform()],
        size: int = 256,
    ) -> None:
        """Initialise Div2k dataset

        Args:
            transforms (list[ImageTransform]):
                List of image transforms to be applied to the input
            path (str, optional): Path to HR Div2k train images.
                Defaults to "data/raw/div2k/train".
            size (int, optional): Size of Target images
                (Raw images will be randomly cropped to be of this size)
        """
        super().__init__(path, transforms, size)
        self.image_list = [
            path
            for path in pathlib.Path(f"{utils.get_project_root()}/{path}").rglob(
                "*.png"
            )
        ]
        self.transforms = transforms
        self.size = size


class Flickr2kDataset(UnsplashDataset):
    """Dataset Class for Flickr2k Datset"""

    def __init__(
        self,
        path: str = "data/raw/flickr2k/Flickr2K_HR",
        transforms: list[transforms.ImageTransform] = [transforms.IdentityTransform()],
        size: int = 256,
    ) -> None:
        """Initialise Flickr2k dataset

        Args:
            transforms (list[ImageTransform]):
                List of image transforms to be applied to the input
            path (str, optional): Path to HR Flickr2k train images.
                Defaults to "data/raw/flickr2k/Flickr2K_HR".
            size (int, optional): Size of Target images
                (Raw images will be randomly cropped to be of this size)
        """
        super().__init__(path, transforms, size)
        self.image_list = [
            path
            for path in pathlib.Path(f"{utils.get_project_root()}/{path}").rglob(
                "*.png"
            )
        ]
        self.transforms = transforms
        self.size = size


class CombinedDataset(UnsplashDataset):
    """Dataset Class that combines multiple Datsets"""

    def __init__(
        self,
        paths: list[str] = [
            "data/raw/unsplash",
            "data/raw/div2k/train",
            "data/raw/flickr2k/Flickr2K_HR",
        ],
        transforms: list[transforms.ImageTransform] = [transforms.IdentityTransform()],
        size: int = 256,
    ) -> None:
        """Combine different data sources

        Args:
            transforms (list[ImageTransform]):
                List of image transforms to be applied to the input
            path (list[str], optional): List of paths to different
            Datasets.
            size (int, optional): Size of Target images
                (Raw images will be randomly cropped to be of this size)
        """
        super().__init__(paths[0], transforms, size)
        file_types = ("jpg", "png")
        self.image_list = []
        for file_type in file_types:
            for dataset_path in paths:
                self.image_list += [
                    path
                    for path in pathlib.Path(
                        f"{utils.get_project_root()}/{dataset_path}"
                    ).rglob(f"*.{file_type}")
                ]
        self.transforms = transforms
        self.size = size
