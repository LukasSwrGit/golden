import torch
import matplotlib.pyplot as plt
import torchvision
import argparse
import timm
import torch.nn as nn 

from train import parse_args
from torchvision.models import resnet18
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = parse_args()

class ResNet18(nn.Module):
    def __init__(self):          
        super(ResNet18, self).__init__()

        self.nr_classes = 10
        self.classif_model = timm.create_model('resnet18', pretrained=args.pretrained)

        #self.classif_model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) #For Black White 1dim channel to function

        self.fc = self.classif_model.fc
        num_ftrs = self.classif_model.fc.in_features           
        self.classif_model.fc = nn.Linear(num_ftrs, self.nr_classes)



    def forward(self, x):
        x = self.classif_model(x)

        return x