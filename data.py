import torch
import matplotlib.pyplot as plt
import torchvision


from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import train
args = train.parse_args()



def load_mnist_loader():
    transform = [
            #transforms.Grayscale(num_output_channels=1),
            transforms.Resize((args.resize_height, args.resize_width)),
            #transforms.Grayscale(3),
            transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
            
            #transforms.Normalize((0.1307,), (0.3081,))
        ]

    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    training_data,  validation_data = torch.utils.data.random_split(training_data, [50000, 10000])

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    trainloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    return trainloader, valloader, testloader

if __name__ == "__main__":

    pass