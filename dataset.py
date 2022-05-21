import torch
import os
from torch.utils.data import Dataset
import PIL.Image as Image

class SRDataset(Dataset):
    def __init__(self, path) -> None:
        self.path = path
        self.image_list = os.listdir(self.path)

    def __getitem__(self, i: int):
        image_path = self.path + '/' + self.image_list[i]
        pil_image = Image.open(image_path)
        return pil_image

