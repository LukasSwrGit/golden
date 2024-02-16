import torch
import matplotlib.pyplot as plt
import torchvision
import argparse
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import skorch
import csv

#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter("runs/mnist1")
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from scipy.stats import loguniform
from datetime import datetime



#These methods were not used for the final experiment but only during until it was found that GridSearchCV etc. were not working well

def return_gridsearchspace():

    search_space = {
        "lr" : [1e-1, 1e-2, 1e-3, 1e-4],
        "optimizer" : [optim.SGD, optim.Adam, optim.Adagrad],
        #"max_epochs" : [25, 50],
        "batch_size" : [32, 64]
    }

    '''search_space = {
        "lr" : [1e-1, 1e-2],
        "optimizer" : [optim.Adam],
    }'''

    return search_space


def grid_search(model, train):
    method = "gridsearch"
    train_iter = iter(train)
    one_batch_train = next(train_iter)
    X, Y = one_batch_train

    grid_model = GridSearchCV(
        estimator=model, 
        param_grid=return_gridsearchspace(), 
        scoring = "accuracy",
        #refit="r2",
        cv=[(slice(None), slice(None))],
        verbose=3
        )
    
    grid_result = grid_model.fit(X, Y)

    # Find the index of the worst performing score
    worst_index = np.argmin(grid_result.cv_results_['mean_test_score'])

    # Extract the worst score and corresponding parameters
    worst_score = grid_result.cv_results_['mean_test_score'][worst_index]
    worst_params = grid_result.cv_results_['params'][worst_index]

    print(f"Worst performers: {worst_index} {worst_score} {worst_params}")

    ###Results:
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #
    best_score = grid_result.best_score_
    best_params = grid_result.best_params_
    #
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    results_txt(method, means, stds, params, best_score, best_params, worst_index, worst_score, worst_params)

    return

def return_randomsearchspace():

    search_space = {
        "lr": loguniform(1e-4, 1e-1),
        "optimizer" : [optim.SGD, optim.Adam, optim.Adagrad],
        #"lr": (1, 0.0001),
        "batch_size" : [32, 64]
    }

    return search_space


def random_search(model, train):
    method = "randomsearch"
    train_iter = iter(train)
    one_batch_train = next(train_iter)
    X, Y = one_batch_train


    random_model = RandomizedSearchCV(
        estimator=model,
        param_distributions=return_randomsearchspace(),
        n_iter=24,  # Number of random combinations to try
        scoring="accuracy",
        #refit="r2",
        cv=[(slice(None), slice(None))],
        verbose=4,
        n_jobs=-1,
        random_state=42
    )

    random_result = random_model.fit(X, Y)

    # Find the index of the worst performing score
    worst_index = np.argmin(random_result.cv_results_['mean_test_score'])

    # Extract the worst score and corresponding parameters
    worst_score = random_result.cv_results_['mean_test_score'][worst_index]
    worst_params = random_result.cv_results_['params'][worst_index]

    # Results:
    print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
    #
    best_score = random_result.best_score_
    best_params = random_result.best_params_
    #
    means = random_result.cv_results_['mean_test_score']
    stds = random_result.cv_results_['std_test_score']
    params = random_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    results_txt(method, means, stds, params, best_score, best_params, worst_index, worst_score, worst_params)

    return


def bayesian_search(model, train):
    method = "bayesiansearch"
    train_iter = iter(train)
    one_batch_train = next(train_iter)
    X, Y = one_batch_train

    search_space = {
        "lr": (1e-4, 1e-2, 'log-uniform'),  # Example values for learning rate in log scale
        #"batch_size": (8, 128),  # Example values for batch size
    }

    bayes_model = BayesSearchCV(
        estimator=model,
        search_spaces=search_space,
        n_iter=10,  # Number of Bayesian optimization steps
        cv=5,
        n_jobs=-1,
        verbose=4,
        random_state=42
    )

    bayes_result = bayes_model.fit(X, Y)

    # Find the index of the worst performing score
    worst_index = np.argmin(bayes_result.cv_results_['mean_test_score'])

    # Extract the worst score and corresponding parameters
    worst_score = bayes_result.cv_results_['mean_test_score'][worst_index]
    worst_params = bayes_result.cv_results_['params'][worst_index]


    # Results:
    print("Best: %f using %s" % (bayes_result.best_score_, bayes_result.best_params_))
    #
    best_score = bayes_result.best_score_
    best_params = bayes_result.best_params_
    #
    means = bayes_result.cv_results_['mean_test_score']
    stds = bayes_result.cv_results_['std_test_score']
    params = bayes_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    results_txt(method, means, stds, params, best_score, best_params, worst_index, worst_score, worst_params)
    
    return


def results_txt(method, means, stds, params, best_score, best_params, worst_index, worst_score, worst_params):

    results = [
        means,
        stds,
        params,
        best_score,
        best_params
    ]

    # Print the results
    for result in results:
        print(f"Saving the following results to a csv: {result}")

    # Specify the CSV file name
    txt_file_path = f"results/{method}_time_{get_current_time()}.txt"

    # Export the results to a CSV file
    with open(txt_file_path, 'w') as txtfile:
        # Write headers
        txtfile.write(f'Overview of the experiment results regarding the conducted {method}:  \n')
        txtfile.write(f'Time of finish: {str(get_current_time())} \n')
        txtfile.write(f'\n')
        txtfile.write(f'\n')
        txtfile.write('\t'.join(['means', 'stds', 'params', 'best_score', 'best_params']) + '\n')

        for result in results:
            if isinstance(result, np.ndarray):
                result_str = '\t'.join(map(str, result))
                txtfile.write('scores: ' + result_str + '\n')
            else:
                # If it's a single float value, you may want to handle it differently
                txtfile.write('best: ' + str(result) + '\n')

        txtfile.write('\n')
        txtfile.write(f'worst index: {worst_index}\n')
        txtfile.write(f'worst score: {worst_score}\n')
        txtfile.write(f'worst params: {worst_params}\n')

    print(f'Results exported to {txt_file_path}')




def get_current_time():
    # Get the current date and time
    current_time = datetime.now()

    # Format the time as a string (optional, you can adjust the format)

    time_str = current_time.strftime("%H_%M_%S")
    return time_str