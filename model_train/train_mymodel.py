import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from PIL import Image
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm

# ====================================================================
class CSVImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = {'bird': 0, 'drone': 1}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 0]
        label = self.label_map[self.annotations.iloc[index, 1]]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# ====================================================================

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        #self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1) 

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        
        self.fc1 = nn.Linear(32 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 2)
        self.bn_fc1 = nn.BatchNorm1d(128)

    def forward(self, x):
        # Assuming input size 224x224
        x = self.pool(F.relu(self.conv1(x)))   # -> 112x112
        x = self.pool(F.relu(self.conv2(x)))   # -> 56x56
        x = self.pool(F.relu(self.conv3(x)))   # -> 28x28
        
        # x = self.pool(F.relu(self.conv4(x)))   # -> 14x14

        x = x.view(x.size(0), -1)              
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.fc2(x)
        return x

# ====================================================================
transform_train = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10),
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                ])
transform_valid = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                ])

train_dataset = CSVImageDataset('labels_train.csv', 'Dataset/train/images', transform=transform_train)
valid_dataset = CSVImageDataset('labels_valid.csv', 'Dataset/valid/images', transform=transform_valid)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================================================================
model_name = 'my_cnn_model'
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
train_loss_history = []
valid_loss_history = []
valid_acc_history = []

print(next(model.parameters()).device)

# ====================================================================

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0,0

    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_loss_history.append(train_loss)

    model.eval()
    val_loss = 0.0
    total, correct = 0.0, 0.0

    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc=f'Epoch {epoch+1} - Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_loss = val_loss / len(valid_loader)
    valid_loss_history.append(valid_loss)
    accuracy = correct / total
    valid_acc_history.append(accuracy)
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {accuracy:.2f}')


os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)

torch.save(model.state_dict(), 'models/00_my_cnn_model.pth')
# ====================================================================
