import os
import pandas as pd

def train():
    image_dir = 'Dataset/train/images'
    labels = []

    for name in os.listdir(image_dir):
        if name.endswith('.jpg'):
            if 'btr' in name.lower():
                label = 'bird'
            elif 'dtr' in name.lower():
                label = 'drone'
            else:
                continue
            labels.append([name, label])

    df = pd.DataFrame(labels, columns=['filename', 'label'])
    df.to_csv('labels_train.csv', index=False)

def valid():
    image_dir = 'Dataset/valid/images'
    labels = []

    for name in os.listdir(image_dir):
        if name.endswith('.jpg'):
            if 'bv' in name.lower():
                label = 'bird'
            elif 'dv' in name.lower():
                label = 'drone'
            else:
                continue
            labels.append([name, label])

    df = pd.DataFrame(labels, columns=['filename', 'label'])
    df.to_csv('labels_valid.csv', index=False)


print('making training csv file')
train()
print('making validation csv file')
valid()
print('process is done')

