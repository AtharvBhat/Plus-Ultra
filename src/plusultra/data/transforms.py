# flake8: noqa
"""
Contains Abstract class and other user defined transforms that get applied to each image
"""

from io import BytesIO
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as ft
from PIL import Image

from plusultra.utils import cv2_to_pil, jpeg_compress_pil, pil_to_cv2


class ImageTransform(object):
    """Base class for all Image transforms"""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, *args, **kwargs) -> Image:
        pass


class IdentityTransform(ImageTransform):
    """Class that does nothing to the inout"""

    def __call__(self, image) -> Image:
        return image


class JpegCorrupt(ImageTransform):
    """
    Applies JPEG corruption to input image
    """

    def __init__(self, prob: float = 1.0, corruption_amount: int = 0) -> None:
        """Initialize a jpeg corruption transform

        Args:
            prob (float, optional): Probability that the input gets compress. Defaults to 1.0.
            corruption_amount (int, optional): Amount of jpeg compression. Defaults to 0.
                If a tuple(low, high) is passed, quality of range is picked randomly within the range
        """
        self.prob = prob
        self.is_range = True if isinstance(corruption_amount, tuple) else False
        self.jpeg_quality = (
            100 - corruption_amount if not self.is_range else corruption_amount
        )

    def __call__(self, image: Image) -> Image:
        if self.is_range:
            low, high = self.jpeg_quality  # type: ignore
            jpeg_quality = np.random.randint(100 - high, 100 - low)  # type: ignore
        else:
            jpeg_quality = self.jpeg_quality
        if np.random.rand() < self.prob:
            return jpeg_compress_pil(image, jpeg_quality)
        else:
            return image


class RandomCrop(ImageTransform):
    """
    Randomly crops an input image to given size
    """

    def __init__(self, size=256) -> None:
        self.size = size

    def __call__(self, image: Image) -> Image:
        cv2_image = pil_to_cv2(image)
        h, w, _ = cv2_image.shape
        assert (
            self.size < h and self.size < w
        ), f"Crop size {self.size} should be smaller than image size {cv2_image.shape}"

        self.randx = torch.randint(0, h - self.size, (1, 1))[0][0]
        self.randy = torch.randint(0, w - self.size, (1, 1))[0][0]

        cv2_image = cv2_image[
            self.randx : self.randx + self.size, self.randy : self.randy + self.size, :
        ]
        image = cv2_to_pil(cv2_image)
        return image

    def apply(self, image: Image) -> Image:
        """Applies stored random crops to a new image

        Args:
            image (Image): Input image

        Returns:
            Image: cropped output
        """
        cv2_image = pil_to_cv2(image)
        cv2_image = cv2_image[
            self.randx : self.randx + self.size, self.randy : self.randy + self.size, :
        ]
        image = cv2_to_pil(cv2_image)
        return image


class PadToMultiple(ImageTransform):
    """
    args : multiple of
    Resize input image leave target as it is.
    Input image width and height will both be resized to nearest multiple of given number
    """

    def __init__(self, multiple):
        self.multiple = multiple

    def __call__(self, image: Image) -> Image:
        cv2_image = pil_to_cv2(image)
        h, w, _ = cv2_image.shape
        h_multiple = (h // self.multiple + 1) * self.multiple
        w_multiple = (w // self.multiple + 1) * self.multiple
        canvas = np.zeros((h_multiple, w_multiple, 3))
        canvas[0:h, 0:w, :] = cv2_image
        return cv2_to_pil(canvas.astype(np.uint8))


class ResizeImage(ImageTransform):
    """
    Resize an input image to a desired multiple
    """

    def __init__(self, scale=0.5):
        self.scale = scale

    def __call__(self, image: Image) -> Any:
        h, w = image.size
        torch_image = ft.resize(
            image, size=(int(self.scale * h), int(self.scale * w)), antialias=True
        )
        return torch_image
