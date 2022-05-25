import torch
import torchvision
import torch.nn.functional as F

class FPN_loss(torch.nn.Module):
    def __init__(self, device) -> None:
        super(FPN_loss, self).__init__()
        self.fpn = torchvision.models.detection.fcos_resnet50_fpn(pretrained=False, pretrained_backbone=True, trainable_backbone_layers=0)
        self.fpn = self.fpn.backbone.body.to(device)

    def forward(self, x, y):
        x_out = self.fpn(x)
        y_out = self.fpn(y)

        x_0, x_1, x_2 = x_out["0"], x_out["1"], x_out["2"]
        y_0, y_1, y_2 = y_out["0"], y_out["1"], y_out["2"]

        loss = F.mse_loss(x_0, y_0) + F.mse_loss(x_1, y_1) + F.mse_loss(x_2, y_2)

        return loss