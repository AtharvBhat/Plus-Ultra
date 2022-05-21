import torch
import os
from torch.utils.data import Dataset
import PIL.Image as Image
import transforms as T

class SRDataset(Dataset):
    def __init__(self, path, transforms=None) -> None:
        self.path = path
        self.image_list = os.listdir(self.path)
        self.transforms = transforms

    def __getitem__(self, i: int):
        image_path = self.path + '/' + self.image_list[i]
        pil_image = Image.open(image_path)
        sample = {"x": pil_image, "y": pil_image}

        if self.transforms is not None:
            return self.transforms(sample)
        else:
            return sample
