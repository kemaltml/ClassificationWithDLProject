import torch
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.mobilenet_v2(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)

model = model.to(device)

state_dict = torch.load('models/mobilenet_v2_TL_model.pth')
model.load_state_dict(state_dict)
model = model.to(device)

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
                                ])

test_data = datasets.ImageFolder('Dataset/test/images/', transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()

model.eval()

all_labels = []
all_predictions = []
test_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

avg_test_loss = test_loss/len(test_loader)

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

model_name = 'mobilenet_v2_TL'
with open(f'results/{model_name}_results.txt', 'w') as f:
    f.write(f'Test Loss: {avg_test_loss:.4f}\n')
    f.write(f'Test Accuracy: {test_accuracy:.4f}\n')
    f.write(f'Precision: {precision:.4f}\n')
    f.write(f'Recall: {recall:.4f}\n')
    f.write(f'F1 Score: {f1:.4f}')

conf_matrix = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(6,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_data.classes, yticklabels=test_data.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'{model_name} Confusion Matrix')
plt.savefig(f'figures/{model_name}_confMatrix.png')

plt.show()
