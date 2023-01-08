"""
This code file is testing the network on greek letters.
We have variations in five different parameters in this code file
Date - 11/26/2022
"""

import sys
import torch
import network
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import glob
import cv2
from torchvision import transforms

# Class definitions
# greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )

# Utility functions
# Function to load the greek letter data set
def create_dataset(train_path, test_path):
    # DataLoader for the Greek data set
    greek_train_folder =  torchvision.datasets.ImageFolder(train_path,
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,))]))
    greek_test_folder = torchvision.datasets.ImageFolder(test_path,
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,))]))

    labels = greek_train_folder.classes

    greek_train = torch.utils.data.DataLoader(greek_train_folder,
        batch_size=15,
        shuffle=True)

    greek_test = torch.utils.data.DataLoader(greek_test_folder,
        batch_size=10,
        shuffle=True)

    return greek_train, greek_test, labels


#Function to load out pretrained network for greek letter testing
def create_model():
    # Create model and load weights of MNIST
    model = network.ConvNet(dropout_rate=0.1)
    model_dict = torch.load('results/model_base.pth')
    model.load_state_dict(model_dict)
    print(model)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace last layer
    model.fc2 = nn.Linear(50, 3)
    print(model)
    return model

#Function to test the network on handwritten greek letters
def test_handwritten_greek_images(model, test_loader, labels):
    test = enumerate(test_loader)
    batch_idx, (X_test, Y_test) = next(test)

    # predict output
    with torch.no_grad():
        output = model(X_test)

    new_data = []
    new_target = []
    for i in range(len(Y_test)):
        new_data.append(X_test[i])
        new_target.append(Y_test[i].item())

    data = torch.stack(new_data)
    for i in range(len(data)):
        print(f'Image: {i + 1} ')
        print(f'Actual Label: {labels[new_target[i]]}')
        print(f'Predicted Label: {labels[output.data.max(1, keepdim=True)[1][i].item()]}')

    # plot the data of first 6
    fig = plt.figure()

    for i in range(len(Y_test)):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(data[i][0], cmap="gray", interpolation="none")
        plt.title("Prediction: {}".format(labels[output.data.max(1, keepdim=True)[1][i].item()]))
        plt.xticks([])
        plt.yticks([])

    print(fig)
    plt.show()

# Main function reads data, trains the model using transfer learning and tests it on handwritten greek letters
def main(argv):
    # Set seed for reproducible results
    random_seed = 2
    torch.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False

    train_path = 'greek_train/'
    handwritten_path = 'handwritten_greek/'
    model = create_model()
    
    # Create data loaders
    greek_train_loader, greek_test_loader, labels = create_dataset(train_path, handwritten_path)
    
    # Train model
    network.train_model(model, greek_train_loader, greek_test_loader, n_epochs=35, save_prefix='_greek', isTest=True)

    # Plot samples from data
    network.plot_samples(greek_train_loader, show_title=False, fig_title="Handwritten letters")
    
    # Test on new images
    test_handwritten_greek_images(model, greek_test_loader, labels)


if __name__ == '__main__':
   main(sys.argv)
