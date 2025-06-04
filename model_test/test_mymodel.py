import torch
import torchvision
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import dataset, transforms, models
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

import os
import sys
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from dataset import ImageFolder

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        #self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1) # This layer is commented out

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
        # self.conv4 and its pooling are commented out
        # x = self.pool(F.relu(self.conv4(x)))   # -> 14x14

        x = x.view(x.size(0), -1)              # Flatten to batch_size * (32 * 28 * 28) = batch_size * 25088
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.fc2(x)
        return x

model_name = '00_my_cnn_model'
device = torch.device('cuda 'if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
model.load_state_dict(torch.load(f'models/{model_name}_model.pth'))
model.eval()

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
])

test_data = dataset.ImageFolder(root='Dataset/test/images', transform=transform)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

criterion = nn.CrossEntropyLoss()

all_labels = []
all_predictions = []
test_loss = 0.0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Testing'):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
test_accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')

print(f'Test Loss: {avg_test_loss:.4f}',
      f'Test Accuracy: {test_accuracy:.4f}',
      f'Precision: {precision:.4f}',
      f'Recall: {recall:.4f}',
      f'F1 Score: {f1:.4f}'
      )

with open(f'results/{model_name}results.txt', 'w') as f:
    f.write(f'Test Loss: {avg_test_loss:.4f}\n',
            f'Test Accuracy: {test_accuracy:.4f}\n',
            f'Precision: {precision:.4f}\n',
            f'Recall: {recall:.4f}\n',
            f'F1 Score: {f1:.4f}'
      )

conf_matrix = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True,
            fmt='d', cmap='Blues',
            xticklabels=test_data.classes, yticklabels=test_data.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(f'figures/{model_name}_confusion_matrix.png')
plt.show()