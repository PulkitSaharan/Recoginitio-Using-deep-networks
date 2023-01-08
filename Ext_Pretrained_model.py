"""
This code file is loading the pretrained network from Torchvision and analyzing the filters in first few Convolution Layers
Date - 11/26/2022
"""

# import statements
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import network

# Creating function to print the filters for particular convolutional layer of a particular model
def print_filters(model,layer):
    # Show filters
    fig = plt.figure()
    for i in range(10):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        #to show filter of particular layer in a particular model
        plt.imshow(model.features[layer].weight[i, 0].detach().numpy())
        plt.title("Fil {}".format(i))
        plt.xticks([])
        plt.yticks([])
    print(fig)
    plt.show()

# Creating function to apply the filter on an image from MNIST dataset and print the filters along with result for particular convolutional layer of a particular model
def apply_filter(model,layer):

    # Show filter effect on first image of train data
    with torch.no_grad():
        train_loader, test_loader = network.load_data()
        train = enumerate(train_loader)
        batch_idx, (X_train, Y_train) = next(train)
        first_img = X_train[0][0].detach().numpy()
        fig = plt.figure()
        plt.imshow(first_img)
        fig.show()
        # taking first 10 filters to plot 20 subplots of filter and application of filter on a digit image
        for i in range(20):
            plt.subplot(5, 4, i + 1)
            plt.tight_layout()
            if (i % 2) == 0:
                filter = model.features[layer].weight[i // 2, 0].detach().numpy()
                plt.imshow(filter, cmap='gray')
                plt.title("Filter: {}".format(i // 2))

            else:
                filter = model.features[layer].weight[(i - 1) // 2, 0].detach().numpy()
                filtered_img = cv2.filter2D(first_img, -1, filter)
                plt.imshow(filtered_img, cmap='gray')
                plt.title("Filtered_img:{}".format((i - 1) // 2))
            plt.xticks([])
            plt.yticks([])
        plt.show()

#main function to run and show the output of filters from pretrained networks
def main():
    torch.manual_seed(42)
    #Select the layer for which you want to see and print the filter and it's application
    layer=0 #2,3
    #load the pretrained model from torchvision
    model = torchvision.models.alexnet(pretrained=True, progress=True)
    #model = torchvision.models.vgg16(pretrained=True, progress=True)
    # Print model
    print(model)
    #Print filter
    print_filters(model,layer)
    #Apply filter
    apply_filter(model,layer)

if __name__ == '__main__':
    main()
