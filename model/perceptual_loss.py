from torchvision import models
import torch.nn as nn

class PerceptualLoss(nn.Module):
    def __init__(self, vgg_model):
        super(PerceptualLoss, self).__init__()
        self.vgg_model = vgg_model
        self.criterion = nn.MSELoss()

    def forward(self, pred, target):
        pred_features = self.vgg_model(pred)
        target_features = self.vgg_model(target)
        loss = self.criterion(pred_features, target_features)
        return loss

# Load a pretrained VGG network
vgg = models.vgg16(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad = False

perceptual_loss = PerceptualLoss(vgg)
