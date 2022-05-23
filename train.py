import torch
from torch.utils.data import DataLoader
import torchvision
import transforms as T
from dataset import SRDataset
import matplotlib.pyplot as plt
from model import AttU_Net2x
from tqdm import tqdm
from utils import train_one_epoch

if __name__ == "__main__":
        
    #hyper params
    batch_size = 8
    lr = 1e-4

    transforms = torchvision.transforms.Compose([T.JpegCorrupt(1, (10, 100)),
                                                T.ToTensor(),
                                                T.RandomCrop(512),
                                                T.ResizeX()])


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    dataset = SRDataset("data/images", transforms=transforms)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    model = AttU_Net2x()
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

    train_one_epoch(model, criterion, train_loader, device, optimizer)
