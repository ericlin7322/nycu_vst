# import your model from net.py
from net import my_network

# from train import CustomImageDataset

from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import pandas as pd
import os
from torchvision.io import read_image

'''
    You can add any other package, class and function if you need.
    You should read the .jpg files located in "./dataset/test/", make predictions based on the weight file "./w_{student_id}.pth", and save the results to "./pred_{student_id}.csv".
'''

# Create a custom dataset class to load images from a specific folder
class CustomImageDataset(ImageFolder):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = sorted(os.listdir(root_dir), key=lambda x: int(x.replace(".jpg", "")))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.file_list[idx])
        image = read_image(img_path).float()

        # Check the number of channels in the image
        if image.shape[0] == 1:
            # If it's a single channel image, duplicate it to make it RGB
            image = torch.cat([image, image, image], dim=0)

        # Convert the torch tensor to a PIL Image
        image = transforms.ToPILImage()(image)

        image = transforms.Compose([
            transforms.ToTensor(),
        ])(image)

        if self.transform:
            image = self.transform(image)

        return image, self.file_list[idx]

def test():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = my_network().to(device) # load model
    model.load_state_dict(torch.load("./w_312551169.pth"))

    criterion = nn.CrossEntropyLoss()

    image_folder = 'dataset/test'

    dataset = CustomImageDataset(root_dir=image_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # testing_data = CustomImageDataset('dataset/custom_test.csv', 'dataset/test')
    # dataloader = DataLoader(testing_data, batch_size=1, shuffle=False)

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    output_file = "./pred_312551169.csv"

    with torch.no_grad(), open(output_file, "w") as file:
        file.write("name,label\n")
        for X, filename in dataloader:
            X = X.to(device)
            pred = model(X)

            predicted_class = pred.argmax(1).item()
            file.write(f"{filename[0]},{predicted_class}\n")

    # with torch.no_grad(), open(output_file, "w") as file:
    #     file.write("name,label\n")
    #     for X, y, filename in dataloader:
    #         X, y = X.to(device), y.to(device)
    #         pred = model(X)

    #         predicted_class = pred.argmax(1).item()
    #         file.write(f"{filename[0]},{predicted_class}\n")

    #         test_loss += criterion(pred, y).item()
    #         correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # test_loss /= num_batches
    # correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__=="__main__":
    test()