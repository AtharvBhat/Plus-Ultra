from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import wandb

def pil_to_cv2(pil_image):
    """
    Given a PIL image object, returns equivalent cv2 image
    """
    np_img = np.array(pil_image)
    cv2_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    return cv2_img

def cv2_to_pil(cv2_img):
    """
    Given a cv2 image, converts it into a PIL image
    """
    np_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(np_img)
    return pil_img

def jpeg_compress(cv2_img, quality):
    """
    Given a cv2 image and a compression quality, compresses the given image using JPEG compression
    """
    _, jpeg_encode = cv2.imencode(".jpg", cv2_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    compressed_img = cv2.imdecode(jpeg_encode, cv2.IMREAD_UNCHANGED)
    return compressed_img

def train_one_epoch(model, criterion, train_loader, device, optimizer, scheduler, epoch, log_step=50):
    """
    Given a model, criterion, dataloader device, optimizer and scheduler
    runs one epoch of training
    """
    avg_loss = 0
    i=0
    for sample in tqdm(train_loader):
        optimizer.zero_grad()
        x, y = sample["x"], sample["y"]
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i%log_step == 0:
            wandb.log({"Step Loss" : loss.item(), "lr" : optimizer.param_groups[0]['lr']})
        avg_loss += loss.item()
        i+=1
    return avg_loss/i
    