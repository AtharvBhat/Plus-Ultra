from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import wandb
import torch
import torch.cuda.amp
import yaml

def get_config(path):
    with open(path, "rb") as f:
        config = yaml.safe_load(f)
    return config

def tensor_to_numpy(image_tensor):
    """
    Given a torch image tensor CxHxW, returns scaled numpy array HxWxC
    which can be displayed by matplotlib or saved using PIL
    """
    unnormalize = torch.clip(image_tensor.cpu().float(), 0, 1) * 255
    permute = unnormalize[0].permute(1,2,0)
    return permute.numpy().astype(np.uint8)


def pil_to_cv2(pil_image):
    """
    Given a PIL image object, returns equivalent cv2 image
    Return : cv2 BGR image
    """
    np_img = np.array(pil_image)
    cv2_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    return cv2_img

def cv2_to_pil(cv2_img):
    """
    Given a cv2 image, converts it into a PIL image
    Return: PIL.Image object
    """
    np_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(np_img)
    return pil_img

def jpeg_compress(cv2_img, quality):
    """
    Given a cv2 image and a compression quality, compresses the given image using JPEG compression
    Return: cv2 BGR image
    """
    _, jpeg_encode = cv2.imencode(".jpg", cv2_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    compressed_img = cv2.imdecode(jpeg_encode, cv2.IMREAD_UNCHANGED)
    return compressed_img

def train_one_epoch(model, criterion, train_loader, device, optimizer, scheduler, epoch, log_step=50, fp_16 = False, config=None):
    """
    Given a model, criterion, dataloader device, optimizer and scheduler
    runs one epoch of training
    Return : Average loss of epoch
    """
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    avg_loss = 0
    i=0
    for sample in tqdm(train_loader):
        optimizer.zero_grad()
        x, y = sample["x"], sample["y"]
        x, y = x.to(device), y.to(device)
        if fp_16:
            with torch.cuda.amp.autocast():   
                y_pred = model(x)
                loss = criterion(y_pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            i+=1
            if i%log_step == 0 and config["wandb"]:
                wandb.log({"Step Loss" : loss.item(), "lr" : optimizer.param_groups[0]['lr']})
            avg_loss += loss.item()
        else:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            i+=1
            if i%log_step == 0 and config["wandb"]:
                wandb.log({"Step Loss" : loss.item(), "lr" : optimizer.param_groups[0]['lr']})
            avg_loss += loss.item()
        
        scheduler.step()
    return avg_loss/i

def run_inference(model, dataloader, device, fp_16=False):
    """
    Given a model and dataloader, run inference on given dataloader
    Return : List of [input, output]
    """
    model.eval()
    model.to(device)
    out_array = []
    for sample in tqdm(dataloader):
        x = sample["x"]
        h, w = sample["h"], sample["w"]
        x = x.to(device)
        with torch.no_grad():
            if fp_16:
                with torch.cuda.amp.autocast():
                    output = model(x)
            else:
                output = model(x)
        output = tensor_to_numpy(output.detach())
        output = output[0:h*2, 0:w*2]
        out_array.append([tensor_to_numpy(sample["x"]), output])

    return out_array

def visualize_outputs(image_list, figsize, path=None):
    """
    Given a list of [input, model output]
    use matpllotlib to visualize results
    optionally, save them to disk
    Return : matplotlib figure
    """
    fig, axes = plt.subplots(len(image_list), 2, figsize=figsize)
    for i, (inp_img, out_img) in enumerate(image_list):
        axes[i][0].imshow(inp_img)
        axes[i][1].imshow(out_img)
        if path is not None:
            pil_image = Image.fromarray(out_img)
            pil_image.save(f"{path}/{i}.png")
    
    return fig

def numParams(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters : {pytorch_total_params:,}")