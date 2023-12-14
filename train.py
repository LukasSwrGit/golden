import torch
import matplotlib.pyplot as plt
import torchvision
import argparse

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--number_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    #parser.add_argument("--train_path", type=str, default="~/Data/") 
    #parser.add_argument("--test_path", type=str, default="~/Data/")
    parser.add_argument("--resize_height", type=int, default=28)
    parser.add_argument("--resize_width", type=int, default=28)
    #parser.add_argument("--model_path", type=str, default=f'/home/lukas/TaskPrediction/task-id-prediction/taskid_pred_models/')  #Best: 50_16 2mistakes
    #parser.add_argument("--nr_test_samples", type=float, default=3)
    #parser.add_argument("--data_from_task", type=float, default=0)
    return parser.parse_args()
args = parse_args()

