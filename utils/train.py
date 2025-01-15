import string
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

from nltk_utils import stem, tokenize, bag_of_words
from model import NeuralNet

# load training data
with open('../assets/intents.json') as file:
    training_dataset = json.load(file)

print(f'Data taining size: {len(training_dataset["intents"])} tags.')

tags = []
all_words = []
xy = []

for intent in training_dataset["intents"]:
    tag = intent["tag"]
    tags.append(tag)

    for speech in intent["speech"]:
        words = tokenize(speech)
        all_words.extend(words)
        xy.append((words, tag))

    if (len(xy) % 5) == 0:
        print(f'Listing dataset: {len(xy)}/{len(training_dataset["intents"])}')

all_words = [stem(word) for word in all_words if word not in string.punctuation]
all_words = sorted(set(all_words)) #sorts cannot have duplicated word so it will remove it all
tags = sorted(set(tags))
print("Dataset listed")

x_train = []
y_train = []
for (pattern, tag) in xy:
    bag = bag_of_words(pattern, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) # CrossEntropyLoss
    if (len(x_train) % 5) == 0:
        print(f'Creating X and Y training {len(x_train)}/{len(xy)}')

x_train = np.array(x_train)
y_train = np.array(y_train)
print("XY training data setted.")

class ChatDataSet(Dataset):
    def __init__(self):
        self.samples_len = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.x_data[index], self.y_data[index]
    
    def __len__(self) -> int:
        return self.samples_len

# Hyperparameters
batch_size = 16
hidden_size = 16
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.0005
num_epochs = 1000
print("Hyperparameters setted")

dataset = ChatDataSet()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print("Everything ready to create the final model")
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words, labels = words.to(device), labels.to(device)

        # forward
        predicted_output = model(words)
        loss = criterion(predicted_output, labels)

        # backwards and optmize step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch: {(epoch + 1)}/{num_epochs}, loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags 
}

FILE = 'model.pth'
torch.save(data, FILE)

print(f'Training complete. File saved to: {FILE}')