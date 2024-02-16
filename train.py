import torch
import matplotlib.pyplot as plt
import torchvision
import argparse
import torch.nn as nn
import numpy as np
import sys
import skorch
import itertools

#import test
 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from hpo_methods import grid_search, random_search, bayesian_search, get_current_time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
batch_size = 32

Gridsearch
adam, 0.001 
sgd, 0.1

Randomsearch
sgd, 0.002175
sgd, 0.021831

'''


def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--number_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, default="adagrad")
    #parser.add_argument("--train_path", type=str, default="~/Data/") 
    #parser.add_argument("--test_path", type=str, default="~/Data/")
    parser.add_argument("--resize_height", type=int, default=32)
    parser.add_argument("--resize_width", type=int, default=32)
    parser.add_argument("--pretrained", type=bool, default=True)
    #parser.add_argument("--model_path", type=str, default=f'/home/lukas/TaskPrediction/task-id-prediction/taskid_pred_models/')  #Best: 50_16 2mistakes
    #parser.add_argument("--nr_test_samples", type=float, default=3)
    #parser.add_argument("--data_from_task", type=float, default=0)
    return parser.parse_args()
args = parse_args()

'''

batch_size: aus data.py dataloader anfordern und batchsize an load_eurosat_loader() übergeben 
lr: beim callen von classifier training lr übergeben
optimizer: beim callen von classifier optimizer übergeben



def grid_search():
        
    
    search_space = {
        "lr" : [1e-1, 1e-2, 1e-3, 1e-4],
        "optimizer" : ["sgd", "adam", "adagrad"],
        "batch_size" : [32, 64]
    }

    lr = 0.001
    optimizer = "sgd"
    batch_size = 32

    model = network.ResNet18()
    train_loader, val_loader, test_loader = data.load_eurosat_loader(batch_size)

    classifier_training(model, train_loader, val_loader, test_loader, optimizer, lr)

    return
'''

def grid_search():

    current_time = get_current_time()
    search_space = {
        "lr": [1e-2, 1e-3, 1e-4],
        "optimizer": ["adam", "sgd"],
        "batch_size": [64, 128]
    }

    # Generate all combinations of hyperparameters
    hyperparameter_combinations = list(itertools.product(search_space['lr'], search_space['optimizer'], search_space['batch_size']))

    best_test_accuracy = 0
    best_hyperparameter_combination = []

    worst_test_accuracy = 1
    worst_hyperparameter_combination = []

    # Loop through each combination
    for lr, optimizer, batch_size in hyperparameter_combinations:
        writer = SummaryWriter(f"runs/training_lr_{lr}_opt_{optimizer}_bs_{batch_size}_time_{current_time}") 

        print(f"Training with lr={lr}, optimizer={optimizer}, batch_size={batch_size}")
        
        # Create a ResNet18 model
        model = network.ResNet18()
        
        # Load data using the current batch size
        train_loader, val_loader, test_loader = data.load_eurosat_loader(batch_size)
        
        # Train the model with the current hyperparameters
        test_accuracy = classifier_training(model, train_loader, val_loader, test_loader, optimizer, lr, writer)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_hyperparameter_combination = f"Best performer: \n lr= {lr} optimizer= {optimizer} batch size= {batch_size} with a test accuracy of {100*test_accuracy:.4f}"
        elif test_accuracy < worst_test_accuracy:
            worst_test_accuracy = test_accuracy
            worst_hyperparameter_combination = f"Worst performer: \n lr= {lr} optimizer= {optimizer} batch size= {batch_size} with a test accuracy of {100*test_accuracy:.4f}"
        
        print("Training completed for this combination.")
        writer.flush()
        writer.close()
    print(best_hyperparameter_combination)
    print(worst_hyperparameter_combination)

    return


def classifier_training(c_model, train_loader, val_loader, test_loader, optimizer, lr, writer):

    training_loss = [] 
    criterion = nn.CrossEntropyLoss()

    #optimizer = torch.optim.Adam(c_model.parameters(), lr=args.lr)
    #optimizer = torch.optim.SGD(c_model.parameters(), lr=args.lr, momentum=0.9) #, weight_decay=0.0001)
    #optimizer = swats.SWATS(c_model.parameters(), lr=args.lr)

    


    #schedulerG = MultiStepLR(optimizer, milestones=[
    #                                200, 250, 290], gamma=0.1, verbose=True)

    c_model.train()
    criterion, c_model = criterion.to(device), c_model.to(device)
    optimizer = return_optimizer(c_model, optimizer, lr)
    epoch = 0
    min_valid_loss = np.inf
    print("Initiating Classifier Training ... ")

    writer.add_scalar('training loss', 2,  epoch)
    writer.add_scalar('training accuracy', 0,  epoch)
    writer.add_scalar('validation loss', 2,  epoch)
    writer.add_scalar('validation accuracy', 0,  epoch)

    for epoch in range(0, args.epoch):
        batch_loss = 0
        mean_batch_loss = 0
        train_loss = 0
        correct = 0
        total = 0
        total_nr_batches = 35                       #careful hardcoded batch nr
        nr_batches = 0
        accuracy = 0

        total_steps = len(train_loader)

        

        print('\nEpoch: %d' % epoch)
        with tqdm(train_loader, unit="batch", desc="Training") as pbar:
            
            losses_batch = []
            for inputs, targets in pbar:
                
                #task_targets = relabel_targets(targets, task)
                #task_targets = targets
                
                
                criterion, c_model, inputs, targets = criterion.to(device), c_model.to(device), inputs.to(device), targets.to(device)

                #optimizer.zero_grad()
                outputs = c_model(inputs)
                #score = outputs.max(dim=1)[0]
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
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

                accuracy = 100 * correct/total
                pbar.set_postfix_str(f"Loss {loss.item():.3f} - Training Accuracy {accuracy:.2f} ")
                pbar.update(1)
                
        writer.add_scalar('training loss', loss,  epoch+1)
        writer.add_scalar('training accuracy', accuracy,  epoch+1)

                
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
            torch.save(c_model.state_dict(), 'temp_model.pt')
            
        #schedulerG.step()
        pbar.close()
        writer.add_scalar('validation loss', (valid_loss / total_val),  epoch+1)
        writer.add_scalar('validation accuracy', val_acc,  epoch+1)
    
    print("-Best model by validation loss-")
    #load_model(model, 'temp_model.pt')
    test_accuracy = test.test(test_loader, c_model, 'temp_model.pt')

    return test_accuracy

def return_optimizer(c_model, optimizer, lr):
    if optimizer == "adam": 
        optimizer = torch.optim.Adam(c_model.parameters(), lr=lr)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(c_model.parameters(), lr=lr)
    elif optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(c_model.parameters(), lr=lr)
    elif optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(c_model.parameters(), lr=lr)
    else:
        print("Error, optimizer not defined.")
    return optimizer


def create_skorch_model():
    model = skorch.NeuralNetClassifier(
        module = network.ResNet18(),
        #max_epochs = 10,
        #batch_size = 100
    )

    return model


def tensorboard_img_grid(train):
    example = iter(train)
    example_data, example_targets = next(example)

    for i in range(2):
        plt.subplot(2,3,i+1)
        plt.imshow(example_data[i][0], cmap='gray')
    
    img_grid = torchvision.utils.make_grid(example_data)
    writer.add_image('mnist_images', img_grid)
   
    writer.add_graph(model, example_data)   #reshape
    writer.close()

    return


if __name__ == "__main__":
    import network
    import data
    import test

    grid_search()
    #train, val, test_loader = data.load_eurosat_loader()
    #model = network.ResNet18()
    #classifier_training(model, train, val, test_loader)
    #tensorboard_img_grid(train)
    #grid_search(create_skorch_model(), data.load_hpo_eurosat_loader())
    #random_search(create_skorch_model(), data.load_hpo_eurosat_loader())
    #bayesian_search(create_skorch_model(), train)

    sys.exit()

    pass
