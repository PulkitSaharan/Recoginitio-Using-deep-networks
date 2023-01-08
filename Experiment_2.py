"""
This code file is experimenting with the existing network.
We have variations in five different parameters together in this code file
Date - 11/26/2022
"""

# import statements
import sys
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
import network
import Experiment

# Main function loads data, creates different models with different parameters and tests on test data
def main():
    # Hyper parameters to be tested
    batch_size_train = [32, 128, 512]
    batch_size_test = 1000
    n_epochs = [3, 5, 7]
    dropout_rate = [0.3, 0.5, 0.7]
    filter_size = [3, 5, 7]
    learning_rate = [0.01,  0.09, 0.2]
    momentum = 0.5
    experiment=1
    for batch_size in batch_size_train:
        train_loader, test_loader = Experiment.load_data_fashion(batch_size)
        for epochs in n_epochs:
            for filter in filter_size:
                for dpr in dropout_rate:
                    for lr in learning_rate:
                        conv_net = network.ConvNet(filter, dpr)
                        #plot_output = f'plots_experiment/{experiment}.png'
                        print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
                        print('Results for experiment %d:' % experiment)
                        print('\tBatch size: {}'.format(batch_size))
                        print('\tEpoch : {}'.format(epochs))
                        print('\tConvulation filter size: {}'.format(filter))
                        print('\tDropout rate: {}'.format(dpr))
                        print('\tLearning rate: {}\n'.format(lr))
                        network.train_model(conv_net, train_loader, test_loader, learning_rate =lr,momentum =0.5,n_epochs = epochs)
                        print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n')
                        experiment+=1

if __name__ == "__main__":
    main()
