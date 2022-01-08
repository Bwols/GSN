from dataset import DataLoader, read_classes
import torch.nn as nn
import hydra

"""
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



