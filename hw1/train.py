# import your model from net.py
from net import my_network
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torchvision.transforms as transforms
from torchvision.io import read_image

import pandas as pd
import os

'''
    You can add any other package, class and function if you need.
    You should read the .jpg from "./dataset/train/" and save your weight to "./w_{student_id}.pth"
'''

class CustomImageDataset(DataLoader):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.filenames = self.img_labels.iloc[:, 0].tolist()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path).float()

        # Check the number of channels in the image
        if image.shape[0] == 1:
            image = torch.cat([image, image, image], dim=0)

        image = transforms.ToPILImage()(image)

        image = transforms.Compose([
            transforms.ToTensor(),
        ])(image)

        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, self.filenames[idx]

def train():

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = my_network().to(device) # load model

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    learning_rate = 0.001
    num_epochs = 150
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    training_data = CustomImageDataset('./dataset/train.csv', './dataset/train')
    train_loader = DataLoader(training_data, batch_size=16  , shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels, _ in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    torch.save(model.state_dict(), './w_312551169.pth')

if __name__ == "__main__":
    train()