"""
File that defines the pytorch dataset class to load data
"""

import pathlib

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

import src.data.transforms as transforms
import src.utils as utils


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
                "*.webp"
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
        image = transforms.RandomCrop(self.size)(image)
        target = image.copy()

        for transform in self.transforms:
            image = transform(image)

        return to_tensor(image), to_tensor(target)
