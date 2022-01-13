from dataset import DataLoader, read_classes
import torch.nn as nn
#https://neptune.ai/blog/pytorch-loss-functions
import torch


import hydra
from omegaconf import DictConfig, OmegaConf

from tests import load_data
from model import PokemonClassifier

from tests import calc_accuracy

calc_accuracy()

"""
batch_size

model = PokemonClassifier()
data = load_data(4,10)
x, labels,_ = data
y = model(x)
print(y.shape[:])
print(labels.shape[:])

criterion = torch.nn.CrossEntropyLoss(reduction='mean')
criterion = torch.nn.NLLLoss()
criterion = torch.nn.BCELoss()
loss = criterion(y,labels)

print(loss)


test_dataloader = DataLoader(batch_size=4, shuffle=True, max_size=400).get_data_loader()
data = iter(test_dataloader).next()


images, labels, class_names = data
print(labels)

from model import PokemonClassifier
model = PokemonClassifier()

output1 = model(images)
print(output1.shape[:])

criterion = nn.CrossEntropyLoss()



loss = criterion(output1, labels)

"""



