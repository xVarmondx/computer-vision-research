import torch.nn as nn
from torchvision import models


def create_model(num_classes):
    """
    Creates a ResNet-18 model and customizes the final layer
    """

    model = models.resnet18(weights='IMAGENET1K_V1')

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model