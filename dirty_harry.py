from dataset import DataLoader, read_classes
import torch.nn as nn
#https://neptune.ai/blog/pytorch-loss-functions
import torch


import hydra
from omegaconf import DictConfig, OmegaConf

from tests import load_data
from model import PokemonClassifier

from tests import calc_accuracy
import tests
#calc_accuracy()

#model = tests.load_model(path_to_model="models/run_0_ResNet152_256.ckpt",architecture="ResNet152",image_size=256)
model = tests.load_model(path_to_model="models/run_0_ResNet152_256.ckpt",architecture="ResNet152",image_size=256)

#data = tests.load_data(dataset_dir="Pokemon_Images_256",labels_csv="pokedex_256.csv" , batch_size=8,max_size=9)

#tests.show_results_of_model(model, data)

#test_dataloader = DataLoader(dataset_dir="Pokemon_Images_256",labels_csv="pokedex_256.csv" ,batch_size=16, shuffle=True, max_size=10).get_data_loader()
#test_dataloader = DataLoader(dataset_dir="Pokemon_Images",labels_csv="pokedex.csv" ,batch_size=16, shuffle=True, max_size=10).get_data_loader()
#calc_accuracy(model=model, test_dataloader=test_dataloader)



model = tests.load_model(path_to_model="models/run_0_ResNet50_64.ckpt",architecture="ResNet50",image_size=64)
#calc_accuracy(model=model, test_dataloader=test_dataloader,device="cuda")
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



