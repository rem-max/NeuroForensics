#!/usr/bin/env python
import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)
num_clients = 20
num_classes = 10
dir_path = "MNIST-noniid-unbalanced-07/"

# Allocate data to users
def generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # Get MNIST data - use existing rawdata from MNIST folder to avoid download
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    # Use existing MNIST rawdata to avoid re-downloading
    existing_rawdata = "MNIST/rawdata"
    trainset = torchvision.datasets.MNIST(
        root=existing_rawdata, train=True, download=False, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=existing_rawdata, train=False, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # Note: alpha is read from dataset_utils.py as a global variable
    # We need to temporarily modify it for this dataset generation
    from utils import dataset_utils
    original_alpha = dataset_utils.alpha
    dataset_utils.alpha = 0.7
    
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid=niid, balance=balance, partition=partition)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
              statistic, niid, balance, partition)
    
    # Restore original alpha
    dataset_utils.alpha = original_alpha


if __name__ == "__main__":
    niid = True      # non-IID
    balance = False  # unbalanced
    partition = 'dir'  # Dirichlet distribution

    generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition)
