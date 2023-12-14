import torch
import matplotlib.pyplot as plt
import torchvision
import argparse
from tqdm import tqdm
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--number_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    #parser.add_argument("--train_path", type=str, default="~/Data/") 
    #parser.add_argument("--test_path", type=str, default="~/Data/")
    parser.add_argument("--resize_height", type=int, default=28)
    parser.add_argument("--resize_width", type=int, default=28)
    parser.add_argument("--pretrained", type=bool, default=True)
    #parser.add_argument("--model_path", type=str, default=f'/home/lukas/TaskPrediction/task-id-prediction/taskid_pred_models/')  #Best: 50_16 2mistakes
    #parser.add_argument("--nr_test_samples", type=float, default=3)
    #parser.add_argument("--data_from_task", type=float, default=0)
    return parser.parse_args()
args = parse_args()

def classifier_training(model, train_loader, val_loader):

    training_loss = []
    c_model = model     
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(c_model.parameters(), lr=args.lr)
    #optimizer = torch.optim.SGD(c_model.parameters(), lr=args.lr, momentum=0.9) #, weight_decay=0.0001)
    #optimizer = swats.SWATS(c_model.parameters(), lr=args.lr)

    #schedulerG = MultiStepLR(optimizer, milestones=[
    #                                200, 250, 290], gamma=0.1, verbose=True)
    
    min_valid_loss = np.inf
    print("Initiating Classifier Training ... ")
    for epoch in range(0, args.epoch):
        batch_loss = 0
        mean_batch_loss = 0
        train_loss = 0
        correct = 0
        total = 0
        total_nr_batches = 35                       #careful hardcoded batch nr
        nr_batches = 0
        print('\nEpoch: %d' % epoch)
        with tqdm(train_loader, unit="batch", desc="Training") as pbar:
            
            losses_batch = []
            for inputs, targets in pbar:
                
                #task_targets = relabel_targets(targets, task)
                #task_targets = targets
                
                c_model.train()
                criterion, c_model, inputs, targets = criterion.to(device), c_model.to(device), inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = c_model(inputs)
                
                #score = outputs.max(dim=1)[0]
                
                loss = criterion(outputs, targets)
                loss.backward()
                
                task_targets = targets
               
                optimizer.step()
                train_loss += loss.item()
                losses_batch.append(loss.item())
                batch_loss = train_loss
                predicted = torch.argmax(outputs, dim=1)
                total += task_targets.size(0)
                correct += predicted.eq(task_targets).sum().item()
                training_loss.append(loss.item())     

                nr_batches += 1
                mean_batch_loss = train_loss / nr_batches
                pbar.set_postfix_str(f"Loss {loss.item():.3f} - Training Accuracy {100*correct/total:.2f} ")
                pbar.update(1)
                
        valid_loss = 0.0
        total_val=0
        c_model.eval()     # Optional when not using Model Specific layer
        for inputs, targets in val_loader:
            #Relabel Targets for loss calculation
            #task_targets = relabel_targets(targets, task)
            task_targets = targets
            inputs, targets = inputs.to(device), targets.to(device)
            
            target = c_model(inputs)
            predicted = torch.argmax(target, dim=1)
            total_val_elem = task_targets.size(0)
            
            task_targets, target = task_targets.to(device), target.to(device)
            correct_val = predicted.eq(task_targets).sum().item()

            loss = criterion(target,task_targets.to(device))
            valid_loss += loss.item() * inputs.size(0)
            total_val += targets.size(0)

        val_acc = 100*correct_val / total_val_elem
        print(f'Epoch {epoch} \t\t Training Loss: {mean_batch_loss:.3f} \t\t Validation Loss: {valid_loss / total_val} \t\t #V-Accuracy#: {val_acc:.3f}')
        
        if min_valid_loss > (loss) :
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{loss:.6f}) \t Saving The Model')
            min_valid_loss = loss
            #not really the minimum val_acc, but the val_acc that can be expected given the min_valid_loss
            min_val_acc = val_acc
            # Saving State Dict
            torch.save(model.state_dict(), 'temp_model.pt')
            
        #schedulerG.step()
        pbar.close()
    
    print("-Best model by validation loss-")
    #load_model(model, 'temp_model.pt')
    return min_val_acc


if __name__ == "__main__":
    import network
    import data

    a, b, c = data.load_mnist_loader()
    model = network.ResNet18()
    
    classifier_training(model, a, b)

    pass