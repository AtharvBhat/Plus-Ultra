"""
File that contains various utils
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml  # type: ignore
from PIL import Image


def get_project_root() -> str:
    """
    Function to return the root dir of the project
    """
    return str(Path(__file__).parents[2])


def get_config(path: str) -> dict[str, Any]:
    """Loads a Yaml config that defines either model or training pipeline parameters
    Args:
        path (str): path to .yaml config file

    Returns:
        dict[str, Any]: Yaml config as python dict
    """
    with open(path, "rb") as f:
        config = yaml.safe_load(f)
    return config


def tensor_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    """Converts a C x H x W Torch image tensor to an H x W x C numpy array

    Args:
        image_tensor (torch.Tensor): Input Image Tensor

    Returns:
        np.ndarray: Output Numpy array
    """
    unnormalize = torch.clip(image_tensor.cpu().float(), 0, 1) * 255
    permute = unnormalize.permute(1, 2, 0)
    return permute.numpy().astype(np.uint8)


def pil_to_cv2(pil_image: Image) -> np.ndarray:
    """Convert PIL Image object to numpy array

    Args:
        pil_image (Image): Input PIL Image Object

    Returns:
        np.ndarray: Output H x W x C numpy array (B G R)
    """
    np_img = np.array(pil_image)
    cv2_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    return cv2_img


def cv2_to_pil(cv2_img: np.ndarray) -> Image:
    """Converts a cv2 image to PIL

    Args:
        cv2_img (np.ndarray): BGR , hxwxc numpy array

    Returns:
        Image: Pil Image Object
    """
    np_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(np_img)
    return pil_img


def jpeg_compress(cv2_img: np.ndarray, quality: int) -> np.ndarray:
    """Injects JPEG compression noise to an input image

    Args:
        cv2_img (np.ndarray): Input BGR, hxwxc numpy array
        quality (int): JPEG compression quality [0-100] lower number is more noise

    Returns:
        np.ndarray: BGR hxwxc np array of image with jpeg noise
    """
    _, jpeg_encode = cv2.imencode(".jpg", cv2_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    compressed_img = cv2.imdecode(jpeg_encode, cv2.IMREAD_UNCHANGED)
    return compressed_img
