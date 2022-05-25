import torch
from torch.utils.data import DataLoader
import torchvision
import transforms as T
from dataset import SRDataset
from model import U_Net2x
from utils import train_one_epoch
from loss_functions import FPN_loss
import torch.nn.functional as F
import wandb

if __name__ == "__main__":
    
    #init wandb
    wandb.init(project="Plus-Ultra", entity="atharvbhat")

    #hyper params
    batch_size = 16
    lr = 1e-3
    num_epochs = 10

    transforms_train = torchvision.transforms.Compose([T.JpegCorrupt(0.5, (10, 100)),
                                                T.ToTensor(),
                                                T.RandomCrop(512),
                                                T.ResizeX()])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataset = SRDataset("data/train_images", transforms=transforms_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)

    model = U_Net2x()
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=num_epochs, steps_per_epoch=len(train_loader))

    wandb.watch(model, criterion, "all", 50, log_graph=True)

    best_loss = None
    for i in range(num_epochs):
        epoch_loss = train_one_epoch(model, criterion, train_loader, device, optimizer, scheduler, i, fp_16=True)
        wandb.log({"Epoch Loss" : epoch_loss, "epoch" : i})
        if best_loss is not None and epoch_loss < best_loss:
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
        else:
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            best_loss = epoch_loss
        

    wandb.finish()