import torch
import matplotlib.pyplot as plt
import torchvision
import argparse
import torch.nn as nn
import numpy as np
import sys
import skorch

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist1")
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from scipy.stats import loguniform


def grid_search(model, train):

    train_iter = iter(train)
    one_batch_train = next(train_iter)
    X, Y = one_batch_train

    search_space = {
        "lr" : [1e-2, 1e-3, 1e-4],
        #"batch_size" : [8, 16, 32, 64, 128]
    }

    grid_model = GridSearchCV(
        estimator=model, 
        param_grid=search_space, 
        scoring = "accuracy",
        #refit="r2",
        cv=[(slice(None), slice(None))],
        verbose=3
        )
    
    grid_result = grid_model.fit(X, Y)

    ###Results:
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))



    return

def random_search(model, train):
    train_iter = iter(train)
    one_batch_train = next(train_iter)
    X, Y = one_batch_train

    search_space = {
        "lr": loguniform(1e-5, 1),
        #"lr": np.logspace(-4, -2, 10),  # Example values for learning rate in log scale
        #"lr": (1, 0.0001),
        #"batch_size": [8, 16, 32, 64, 128]
    }

    random_model = RandomizedSearchCV(
        estimator=model,
        param_distributions=search_space,
        n_iter=10,  # Number of random combinations to try
        #scoring=["accuracy"],
        #refit="r2",
        cv=[(slice(None), slice(None))],
        verbose=4,
        n_jobs=-1,
        random_state=42
    )

    random_result = random_model.fit(X, Y)

    # Results:
    print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
    means = random_result.cv_results_['mean_test_score']
    stds = random_result.cv_results_['std_test_score']
    params = random_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return


def bayesian_search(model, train):
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

    # Results:
    print("Best: %f using %s" % (bayes_result.best_score_, bayes_result.best_params_))
    means = bayes_result.cv_results_['mean_test_score']
    stds = bayes_result.cv_results_['std_test_score']
    params = bayes_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return