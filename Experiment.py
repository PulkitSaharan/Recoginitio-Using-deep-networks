"""
This code file is experimenting with the existing network.
We have variations in five different parameters in this code file
Date - 11/26/2022
"""
# import statements
import sys
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import Dataset
import network

#Loading data from Fashin MNSIT data set
def load_data_fashion(batch_size_train=64, batch_size_test=1000):
    # Load training data
    train_data = datasets.FashionMNIST(
        root="fashion_data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])
    )
    # Load test data
    test_data = datasets.FashionMNIST(
        root="fashion_data",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])
    )
    # Create a train data loader
    train_loader = torch.utils.data.DataLoader(
        train_data,
        shuffle=True,
        batch_size=batch_size_train
    )
    # Create a test data loader
    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=True,
        batch_size=batch_size_test
    )
    return train_loader, test_loader


# Main function creates many different models with different hyperparameters and conducts a linear search to see effect of
# each parameter individually
#Default hyperparameter values - lr = 0.1, epoch =5, batch_size=64, dpr=0.5, filter size =5
def main():
    #Different values of different parameters
    batch_size_train=[32,64,128,256,512]
    batch_size_test=1000
    n_epochs = np.linspace(1, 20,num=20,dtype=int)
    dropout_rate=np.linspace(0.01, 0.9,num=20,dtype=float)
    filter_size=[1,3,5,7,9]
    learning_rate= np.linspace(0.001, 0.3,num=20,dtype=float)
    momentum= 0.5

   #Default Paramters
    lr = 0.01
    batch_size=64
    epochs=5
    filter=5
    dpr=0.5

#    # variation in dropout rate
    dropout_list=[]
    test_accuracy_dpr=[]
    train_loader, test_loader = load_data_fashion(batch_size)
    fig=plt.figure()
    for dpr in dropout_rate:
        print('Dropout Rate : {}'.format(dpr))
        conv_net = network.ConvNet(filter,dpr)
        acc=network.train_model(conv_net, train_loader, test_loader,learning_rate= lr,momentum=momentum,n_epochs=epochs, save_prefix = '_base')
        dropout_list.append(dpr)
        test_accuracy_dpr.append(acc)
    plt.plot(dropout_list,test_accuracy_dpr, color='blue')
    plt.title('Dropout Rate vs Test Accuracy')
    plt.xlabel('Droptout Rate')
    plt.ylabel('Test Accuracy')
    plt.savefig('dpr_acc.png')
#
##    # variation in epochs
    Epochs_list = []
    test_accuracy_epochs = []
    train_loader, test_loader = load_data_fashion(batch_size)
    fig = plt.figure()
    for epochs in n_epochs:
        print('Epoch number : {}'.format(epochs))
        conv_net = network.ConvNet(filter,dpr)
        acc = network.train_model(conv_net, train_loader, test_loader, learning_rate= lr,momentum=momentum,n_epochs=epochs, save_prefix = '_base')
        Epochs_list.append(epochs)
        test_accuracy_epochs.append(acc)
    plt.plot(Epochs_list, test_accuracy_epochs, color='red')
    plt.title('Number of Epochs vs Test Accuracy')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Test Accuracy')
    plt.savefig('epoch_acc.png')
#
##    # variation in batch size
    batch_size_list = []
    test_accuracy_batch= []
    fig = plt.figure()
    for batch in batch_size_train:
        train_loader, test_loader = load_data_fashion(batch)
        print('Batch size for train data : {}'.format(batch))
        conv_net = network.ConvNet(filter, dpr)
        acc = network.train_model(conv_net, train_loader, test_loader, learning_rate= lr,momentum=momentum,n_epochs=epochs, save_prefix = '_base')
        batch_size_list.append(batch)
        test_accuracy_batch.append(acc)
    plt.plot(batch_size_list, test_accuracy_batch, color='yellow')
    plt.title('Train Batch Size vs Test Accuracy')
    plt.xlabel('Batch Size')
    plt.ylabel('Test Accuracy')
    plt.savefig('batch_size_acc.png')

#    # variation in filter size
    filter_list = []
    test_accuracy_filter = []
    train_loader, test_loader = load_data_fashion(batch_size)
    fig = plt.figure()
    for filter in filter_size:
        print('Filter size : {}'.format(filter))
        conv_net = network.ConvNet(filter, dpr)
        acc = network.train_model(conv_net, train_loader, test_loader,learning_rate= lr,momentum=momentum,n_epochs=epochs, save_prefix='_base')
        filter_list.append(filter)
        test_accuracy_filter.append(acc)
    plt.plot(filter_list, test_accuracy_filter, color='green')
    plt.title('Filter Size vs Test Accuracy')
    plt.xlabel('Size of Filter')
    plt.ylabel('Test Accuracy')
    plt.savefig('filter_acc.png')

  #  # variation in learning rate
    lr_list = []
    test_accuracy_lr = []
    train_loader, test_loader = load_data_fashion(batch_size)
    fig = plt.figure()
    for lr in learning_rate:
        print('Learnign Rate : {}'.format(lr))
        conv_net = network.ConvNet(filter, dpr)
        acc = network.train_model(conv_net, train_loader, test_loader, learning_rate= lr,momentum=momentum,n_epochs=epochs, save_prefix='_base')
        lr_list.append(lr)
        test_accuracy_lr.append(acc)
    plt.plot(lr_list, test_accuracy_lr, color='orange')
    plt.title('Learning Rate vs Test Accuracy')
    plt.xlabel('Learning Rate')
    plt.ylabel('Test Accuracy')
    plt.savefig('lr_acc.png')

if __name__ == "__main__":
    main()
