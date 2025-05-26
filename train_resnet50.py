import os
import pandas as pd 
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models 
from torchvision.models import ResNet50_Weights, resnet50
from PIL import Image
import torch.nn as nn
import torch.optim as optim 
from matplotlib import pyplot as plt 
from tqdm import tqdm
import shutil

class CSVImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform 
        self.label_map = {'bird':0, 'drone':1}

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

transform = transforms.Compose([transforms.Resize((224,224)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
                                ])
train_dataset = CSVImageDataset('labels_train.csv', 'Dataset/train/images', transform=transform)
valid_dataset = CSVImageDataset('labels_valid.csv', 'Dataset/valid/images', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'resnet50'
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 10
train_loss_history = []
valid_loss_history = []
valid_acc_history = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0 
    
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training'):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_loss_history.append(train_loss)

    model.eval()
    val_loss = 0.0 
    correct = 0 
    total = 0 

    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc=f'Epoch {epoch+1}/{epochs} - Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    valid_loss = val_loss / len(valid_loader)
    valid_loss_history.append(valid_loss)
    accuracy = correct / total 
    valid_acc_history.append(accuracy)

    print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | Accuracy: {accuracy:.4f}\n')

torch.save(model.state_dict(), f'models/{model_name}_model.pth')


plt.plot(train_loss_history, label='Train Loss')
plt.plot(valid_loss_history, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0,1)
plt.legend()
plt.title(f'{model_name} Loss Graph')
plt.savefig(f'figures/{model_name}_loss.png')

plt.plot(valid_acc_history, label = 'Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.legend()
plt.title(f'{model_name} Validation Accuracy')
plt.savefig(f'figures/{model_name}_valid.png')
plt.show()


