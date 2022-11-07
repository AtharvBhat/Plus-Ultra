import math
import torch
from torch.utils.data import DataLoader
import torchvision
import transforms as T
from dataset import SRDataset
from model import Unet
from utils import get_config, train_one_epoch, numParams
from loss_functions import FPN_loss
import torch.nn.functional as F
import wandb
import kornia

if __name__ == "__main__":

    config = get_config("config.yaml")
    
    if config["wandb"]:
        #init wandb
        wandb.init(project="Plus-Ultra", entity="atharvbhat", config=config)

    #hyper params
    batch_size = config["batch_size"]
    lr = config["lr"]
    num_epochs = config["epochs"]
    weight_decay = config["weight_decay"]
    fp_16 = config["fp_16"]
    train_data_path = config["train_path"]

    transforms_train = torchvision.transforms.Compose([T.JpegCorrupt(0.5, (50, 70)),
                                                T.ToTensor(),
                                                T.RandomCrop(512),
                                                T.ResizeX()])

    device = torch.device('cuda') if torch.cuda.is_available() and config["device"] == "cuda" else torch.device('cpu')

    train_dataset = SRDataset(train_data_path, transforms=transforms_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)

    model = Unet(config=config)
    numParams(model)
    model = model.to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, pct_start=0.1, anneal_strategy='linear', epochs=num_epochs, steps_per_epoch=len(train_loader))

    #load weights if finetuning
    #model.load_state_dict(torch.load("checkpoints/best_model.pth")["weights"])
    if config["wandb"]:
        wandb.watch(model, criterion, "all", 50, log_graph=False)

    best_loss = None
    for i in range(num_epochs):
        epoch_loss = train_one_epoch(model, criterion, train_loader, device, optimizer, scheduler, i, fp_16=fp_16, config=config)
        if config["wandb"]:
            wandb.log({"Epoch Loss" : epoch_loss, "epoch" : i})
        if best_loss != None and epoch_loss < best_loss:
            torch.save({"weights":model.state_dict(), "config" : config}, "checkpoints/best_model.pth")
        if best_loss == None :
            torch.save({"weights":model.state_dict(), "config" : config}, "checkpoints/best_model.pth")
            best_loss = epoch_loss
        
    if config["wandb"]:
        wandb.finish()