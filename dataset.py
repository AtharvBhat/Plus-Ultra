import os
import torch
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
        pil_image = Image.open(image_path).convert('RGB')
        sample = {"x": pil_image, "y": pil_image}

        if self.transforms is not None:
                return self.transforms(sample)
        else:
            print("Dataset Object needs transforms!")

    def __len__(self):
        return len(self.image_list)

