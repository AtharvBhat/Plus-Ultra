import torch
from torch.utils.data import DataLoader
import torchvision
import transforms as T
from dataset import SRDataset
from model import Unet
from utils import run_inference, visualize_outputs, get_config
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
    
    config = get_config("config.yaml")
    test_data_path = config["test_path"]

    transforms_test = torchvision.transforms.Compose([T.ToTensor(),
                                                    T.PadToMultiple(16)])


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_dataset = SRDataset(test_data_path, transforms=transforms_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    model = Unet()
    model = model.to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))

    test_outs = run_inference(model, test_loader, device, fp_16=True)
    fig = visualize_outputs(test_outs, (15,10), "data/outputs")
    plt.show()
    