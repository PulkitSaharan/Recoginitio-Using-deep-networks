"""
This code file is testing the network on the first 10 images of test test and on the handwritten digits from 0 to 9.
We have variations in five different parameters in this code file
Date - 11/22/2022
"""
# import statements
import sys
import torch
import network
import matplotlib.pyplot as plt
import glob
import cv2
import torchvision
from torchvision import transforms


# Function preprocesses handwritten input image and converts it to required binary format
def process_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (7, 7), 0)
    ret, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY_INV +
                                            cv2.THRESH_OTSU)
    image = cv2.resize(image, (28, 28))
    return image

#Function to read and transform the handwritten digits
def test_handwritten_images(model):
    images_tensor_list = []
    images_list = []
    for filename in glob.glob('handwritten_digits/*.jpg'):
        print(filename)
        image = process_image(filename)
        images_list.append(image)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])
        img_transform = transform(image)
        images_tensor_list.append(img_transform)

    images = torch.stack(images_tensor_list)

    # predict output
    with torch.no_grad():
        output = model(images)

    # plot results
    fig = plt.figure()
    for i in range(10):
        plt.subplot(4, 3, i + 1)
        plt.tight_layout()
        plt.imshow(images_list[i], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])

    plt.show()

#Function to test the network on first 10 digits of test data set
def test_10_images(model):
    train_loader, test_loader = network.load_data()

    test = enumerate(test_loader)
    batch_idx, (X_test, Y_test) = next(test)
    
    # predict output
    with torch.no_grad():
        output = model(X_test)

    ten_data = []
    ten_target = []
    for i in range(10):
        ten_data.append(X_test[i])
        ten_target.append(Y_test[i].item())

    # print predictions
    data = torch.stack(ten_data)
    for i in range(len(data)):
        print(f'Image: {i + 1} ')
        print(f'Actual Label: {ten_target[i]}')
        print(f'Predicted Label: {output.data.max(1, keepdim=True)[1][i].item()}')

    # plot the data of first 9
    fig = plt.figure()

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(data[i][0], cmap="gray", interpolation="none")
        plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])

    print(fig)
    plt.show()

# Main function creates a network and loads pre trained weights, tests it on test set and new handwritten images
def main(argv):
    # load network
    model = network.ConvNet()
    model_dict = torch.load('results/model_base.pth')
    model.load_state_dict(model_dict)
    model.eval()
    # test network on new images
    test_10_images(model)
    test_handwritten_images(model)


if __name__ == '__main__':
   main(sys.argv)
