import torchvision.transforms.functional as T
import torch
import numpy as np
from utils import pil_to_cv2, cv2_to_pil, jpeg_compress

class JpegCorrupt(object):
    """
    args:
    prob : probability of applying jpeg compression
    range : (low, high) range of jpeg quality to randomly select amount of compression
    """
    def __init__(self, prob, range) -> None:
        self.prob = prob
        self.low, self.high = range

    def __call__(self, sample):
        x, y = sample["x"], sample["y"]

        #jpeg compress x   
        if np.random.rand() < self.prob:
            cv2_image = pil_to_cv2(x)
            jpeg_quality = np.random.randint(self.low, self.high)
            cv2_image_compressed = jpeg_compress(cv2_image, jpeg_quality)
            x = cv2_to_pil(cv2_image_compressed)
        
        return {"x": x, "y": y}


class RandomCrop(object):
    """
    args:
    size : height and width of the final crop given an input image
    """
    def __init__(self, size=256) -> None:
        self.size = size

    def __call__(self, sample):
        x = sample["x"]
        y = sample["y"]
        _, h ,w = x.shape

        assert x.shape == y.shape , "Both input and target should be same size"
        assert self.size < h and self.size < w , f"Crop size {self.size} should be smaller than image size {x.shape}"

        randx = torch.randint(0, h-self.size, (1,1))[0][0]
        randy = torch.randint(0, w-self.size, (1,1))[0][0]

        x_crop = x[:, randx: randx + self.size, randy: randy + self.size]
        y_crop = y[:, randx: randx + self.size, randy: randy + self.size]

        return {"x": x_crop, "y":y_crop}

class ToTensor(object):
    """
    Convert input image pairs to tensors
    """
    def __init__(self) -> None:
        pass
    def __call__(self, sample):
        x = sample["x"]
        y = sample["y"]

        x_tensor = T.to_tensor(x)
        y_tensor = T.to_tensor(y)

        return {"x": x_tensor, "y":y_tensor}

class ResizeX(object):
    """
    args : scale (how much to rescale input image default 0.5)
    Resize input image leave target as it is
    """
    def __init__(self, scale=0.5):
        self.scale = scale

    def __call__(self, sample):
        x, y = sample["x"], sample["y"]
        _, h, w = x.shape
        x = T.resize(x, size=(int(self.scale*h), int(self.scale*w)))
        return {"x":x, "y":y}

class PadToMultiple(object):
    """
    args : multiple of
    Resize input image leave target as it is.
    Input image width and height will both be resized to nearest multiple of given number
    """
    def __init__(self, multiple):
        self.multiple = multiple

    def __call__(self, sample):
        x, y = sample["x"], sample["y"]
        _, h, w = x.shape
        h_multiple = (h//self.multiple + 1) * self.multiple
        w_multiple = (w//self.multiple + 1) * self.multiple
        canvas = torch.zeros((3, h_multiple, w_multiple))
        canvas[:, 0:h, 0:w] = x
        return {"x":canvas, "y":y, "h":h, "w":w}