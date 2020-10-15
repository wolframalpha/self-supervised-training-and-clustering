import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self, features_dim=128):
        super(Model, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = torch. nn.Linear(512, 256)
        self.backbone_dim = self.backbone.fc.out_features
        self.head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.ReLU(), nn.Linear(self.backbone_dim, features_dim))
 
    def forward(self, x):
        features = self.head(self.backbone(x))
        features = F.normalize(features, dim = -1)
        return features
    
