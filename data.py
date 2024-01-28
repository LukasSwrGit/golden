import torch
import matplotlib.pyplot as plt
import torchvision
import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import train
args = train.parse_args()



def load_mnist_loader():
    transformations = transforms.Compose([
            #transforms.Grayscale(num_output_channels=1),
            transforms.Resize((args.resize_height, args.resize_width)),
            #transforms.Grayscale(3),
            transforms.ToTensor(),
            #torchvision.transforms.Normalize((0.5,), (0.5,))
            
            transforms.Normalize((0.5,), (0.5,))
        ])

    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transformations
    )
    training_data,  validation_data = torch.utils.data.random_split(training_data, [50000, 10000])

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transformations
    )

    trainloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    return trainloader, valloader, testloader

def load_eurosat_loader():
    transformations = transforms.Compose([
            #transforms.Grayscale(num_output_channels=1),
            transforms.Resize((args.resize_height, args.resize_width)),
            #transforms.Grayscale(3),
            transforms.ToTensor(),
            #torchvision.transforms.Normalize((0.5,), (0.5,))
            
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    training_data = datasets.EuroSAT(
        root="data",
        transform=transformations,
        download=True,
    )
    
    training_data,  test_data = torch.utils.data.random_split(training_data, [24000, 3000])
    training_data,  validation_data = torch.utils.data.random_split(training_data, [20000, 4000])

    trainloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    return trainloader, valloader, testloader


def load_hpo_eurosat_loader():
    transformations = transforms.Compose([
            #transforms.Grayscale(num_output_channels=1),
            transforms.Resize((args.resize_height, args.resize_width)),
            #transforms.Grayscale(3),
            transforms.ToTensor(),
            #torchvision.transforms.Normalize((0.5,), (0.5,)) 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        ])

    training_data = datasets.EuroSAT(
        root="data",
        transform=transformations,
        download=True,
    )
    
    #training_data,  test_data = torch.utils.data.random_split(training_data, [24000, 3000])
    #training_data,  validation_data = torch.utils.data.random_split(training_data, [20000, 4000])

    trainloader = DataLoader(training_data, batch_size=len(training_data), shuffle=True)
    #valloader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=True)
    #testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    return trainloader

if __name__ == "__main__":

    pass