import torchvision.transforms.functional as T
import torch

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
        assert self.size < h or self.size < w , f"Crop size {self.size} should be smaller than image size {x.shape}"

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