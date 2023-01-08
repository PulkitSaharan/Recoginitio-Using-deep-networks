"""
This code file is loading the pretrained our network from results and analyzing the filters in first  Convolution Layer
Date - 11/26/2022
"""
# import statements
import torch
import matplotlib.pyplot as plt
import cv2
import network


# Creating function to print the filters for  convolutional layer1 of a particular model
def print_filters(model):
    # Show filters
    fig = plt.figure()
    for i in range(10):
        print(f'Filter: {i + 1}')
        print(f'Current Filter Shape: {model.conv1.weight[i, 0].shape}')
        print(f'Current Filter Weight: {model.conv1.weight[i, 0]}')
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        # to show filter of particular layer in a particular model
        plt.imshow(model.conv1.weight[i, 0].detach().numpy())
        plt.title("Filter: {}".format(i))
        plt.xticks([])
        plt.yticks([])
    print(fig)
    plt.show()

# Creating function to apply the filter on an image from MNIST dataset and print the filters along with result for  convolutional layer1 of a particular model
def apply_filter(model):

# Show filter effect on first image of train data
    with torch.no_grad():
        train_loader, test_loader = network.load_data()
        train = enumerate(train_loader)
        batch_idx, (X_train, Y_train) = next(train)
        first_img = X_train[0][0].detach().numpy()
        fig = plt.figure()
        plt.imshow(first_img)
        fig.show()
#        for i in range(10):
#            filter = model.conv1.weight[i, 0].detach().numpy()
#            filtered_img= cv2.filter2D(first_img,-1, filter)
#            plt.subplot(4, 5, i + 1)
#            plt.tight_layout()
#            plt.imshow(filter, cmap = 'gray')
#            plt.title("Filter: {}".format(i+1))
#            plt.xticks([])
#            plt.yticks([])
#
#            plt.subplot(4, 5, i + 11)
#            plt.tight_layout()
#            plt.imshow(filtered_img, cmap = 'gray')
#            plt.title("Filtered_img:{}".format(i+1))
#            plt.xticks([])
#            plt.yticks([])
#        plt.show()

        for i in range(20):
            plt.subplot(5, 4, i + 1)
            plt.tight_layout()
            if (i%2)==0:
                filter = model.conv1.weight[i//2, 0].detach().numpy()
                plt.imshow(filter, cmap = 'gray')
                plt.title("Filter: {}".format(i//2))

            else:
                filter = model.conv1.weight[(i-1)//2, 0].detach().numpy()
                filtered_img = cv2.filter2D(first_img, -1, filter)
                plt.imshow(filtered_img, cmap = 'gray')
                plt.title("Filtered_img:{}".format((i-1)//2))
            plt.xticks([])
            plt.yticks([])
        plt.show()

        
# Main function loads a network and prints filters in convolutional layer and image convoled with these filters
def main():
    model = network.ConvNet(5,0.5)
    model_dict = torch.load('results/model_base.pth')
    model.load_state_dict(model_dict)

    # Print model
    print(model)
    print_filters(model)
    apply_filter(model)

if __name__ == '__main__':
    main()
