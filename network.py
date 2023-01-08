"""
This code file includes functions to build the convolution network, load the data, plotting the graphs, training and testing the model
Date - 11/21/2022
"""
import torch
import torchvision
import cv2
import sys
from torchvision import datasets
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# # Create a convolutional neural network class
class ConvNet(nn.Module):
    def __init__(self,filter_size=5,dropout_rate=0.5):
        super(ConvNet, self).__init__()
        # A convolution layer with 10 5x5 filters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=filter_size)
        # A convolution layer with 20 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=filter_size)
        # A dropout layer with a 0.5 dropout rate (50%)
        self.conv2_dropout = nn.Dropout2d(dropout_rate)
        half_filter = filter_size // 2
        output_size = ((28 - 2 * half_filter) // 2 - 2 * half_filter) // 2
        self.fc1 = nn.Linear(20*output_size*output_size, 50)
        # A final fully connected Linear layer with 10 nodes
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # A convolution layer with 10 5x5 filters
        x = self.conv1(x)
        # A max pooling layer with a 2x2 window and a ReLU function applied.
        x = F.relu(F.max_pool2d(x, (2, 2)))
        x = self.conv2(x)
        x = self.conv2_dropout(x)
        # A max pooling layer with a 2x2 window and a ReLU function applied
        x = F.relu(F.max_pool2d(x, (2, 2)))
        # Flattening Operation
        x = torch.flatten(x, start_dim=1)
        # ReLU function on the output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # the log_softmax function applied to the output.
        return F.log_softmax(x)

# Function to load data and return data loaders
def load_data(batch_size_train=64, batch_size_test=1000):
    # Load training data
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])
    )
    # Load test data
    test_data = datasets.MNIST(
        root="data",
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


# Function plots first 6 digits from data loader
def plot_samples(data_loader, show_title = True, fig_title = 'Figure'):
    examples = enumerate(data_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)

    fig = plt.figure(fig_title)
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray')
        if show_title:
            plt.title("Label: {}".format(example_targets[i]))
        plt.axis("off")
    plt.show()


# Train loop for one epoch
def train(network, dataloader, epoch, optimizer, train_losses, train_counter, log_interval):
    network.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                       100. * batch_idx / len(dataloader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(dataloader.dataset)))
            torch.save(network.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')


# Test loop for one epoch
def test_network(network, test_loader, test_losses):
    network.eval()
    test_loss= 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss +=F.nll_loss(output, target,size_average= False ).item()
            pred =output.data.max(1,keepdim=True)[1]
            correct +=pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return correct / len(test_loader.dataset)
    
    
# Plot train and test performance of model
def plot_performance(train_counter, train_losses, test_counter, test_losses, isTest = True):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    if isTest:
        plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train loss', 'Test loss'], loc = 'upper right')
    plt.xlabel('# training examples')
    plt.ylabel('negative log likelihood loss')
    plt.show()


# Function to train model for n epochs and plot performance
def train_model(conv_net, train_loader, test_loader, learning_rate = 0.01,momentum =0.5,n_epochs = 3, save_prefix = '_base', isTest = True):
    batch_size_train = 64
    batch_size_test = 1000
    log_interval = 10
    train_losses = []
    train_counter = []
    test_losses = []

    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
    optimizer = optim.SGD(conv_net.parameters(), lr=learning_rate,
                                                momentum=momentum)
    # Run training
    if isTest:
        acc=test_network(conv_net, test_loader, test_losses)
    for epoch in range(1, n_epochs+1):
        train(conv_net, train_loader, epoch, optimizer, train_losses, train_counter, log_interval)
        if isTest:
            acc=test_network(conv_net, test_loader, test_losses)

    # Save model files
    torch.save(conv_net.state_dict(), 'results/model'+save_prefix+'.pth')
    torch.save(optimizer.state_dict(), 'results/optimizer'+save_prefix+'.pth')

    # Plot performance
    plot_performance(train_counter, train_losses, test_counter, test_losses, isTest)
    if isTest:
        return acc.item()
    return 0


# Main function that loads data, creates a neural network and trains it
def main(argv):

    # Setting random seed and disabling cudnn
    random_seed = 42
    learning_rate = 0.01  # learning rate
    momentum = 0.5  # refers to inertia
    n_epochs = 3
    torch.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False

    train_loader, test_loader = load_data()

    # Plot first 6 digits
    plot_samples(train_loader)
    conv_net = ConvNet()
    print(conv_net)

    # model training and testing and plotting the training curve
    train_model(conv_net, train_loader, test_loader, learning_rate=learning_rate, momentum=momentum, n_epochs=n_epochs)


if __name__ == '__main__':
    main(sys.argv)
