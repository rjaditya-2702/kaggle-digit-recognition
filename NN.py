import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd

'''
    CODE TO PREPARE DATA.
'''
train_path = '/Users/rjaditya/Documents/projects/Kaggle/digit/dataset/train.csv'
test_path = '/Users/rjaditya/Documents/projects/Kaggle/digit/dataset/test.csv'

dataset = pd.read_csv(train_path, header=0)
submission_dataset = pd.read_csv(test_path, header=0)

y = dataset['label']
X = dataset.drop(labels='label')

x_train, x_test, y_train, y_test = train_test_split(
  X,y , random_state=104,test_size=0.25, shuffle=True)

trainloader = torch.utils.data.DataLoader([x_train, y_train], batch_size=40,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader([x_test, y_test], batch_size=40,
                                         shuffle=False, num_workers=2)

model = nn.Sequential(
    nn.Linear(in_features=783, out_features=500),
    nn.ReLU(),
    nn.Linear(in_features=500, out_features=250),
    nn.ReLU(),
    nn.Linear(in_features=250, out_features=10),
    nn.Softmax()
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
loss = nn.BCELoss
training_loss = []
for epoch in range(200):  # loop over the dataset multiple times
    running_loss = 0
    for x_train_example, y_train_example in zip(x_train, y_train):
        optimizer.zero_grad()
        y_hat = model(x_train_example)
        tloss = loss(y_hat, y_train_example)
        tloss.backward()
        optimizer.step()
        running_loss += tloss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    