import torch
import torchvision
import transforms as T
from dataset import SRDataset
import matplotlib.pyplot as plt


transforms = torchvision.transforms.Compose([T.ToTensor(),
                                            T.RandomCrop(256)])


data = SRDataset("data/images", transforms=transforms)

sample = data[200]
print(sample["x"].shape)
plt.imshow(sample["x"].permute(1, 2, 0))
plt.show()
plt.imshow(sample["y"].permute(1, 2, 0))
plt.show()