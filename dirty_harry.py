from dataset import DataLoader, read_classes
import torch.nn as nn

import torch


import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="config",config_name="default_config")
def use_hydra(cfg):
    print(OmegaConf.to_yaml(cfg))


"""
batch_size


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



