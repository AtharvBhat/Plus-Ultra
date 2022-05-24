import torch
from torch.utils.data import DataLoader
import torchvision
import transforms as T
from dataset import SRDataset
from model import AttU_Net2x
from utils import run_inference, visualize_outputs
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    transforms_test = torchvision.transforms.Compose([T.ToTensor(),
                                                    T.PadToMultiple(16)])


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_dataset = SRDataset("data/test_images", transforms=transforms_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    model = AttU_Net2x()
    model = model.to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))

    test_outs = run_inference(model, test_loader, device)
    fig = visualize_outputs(test_outs, (20,10), "data/outputs")
    plt.show()
    