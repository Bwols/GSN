import torch
import csv
import numpy
import cv2
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.utils as utils
import numpy as np
from utils import accuracy
from model import PokemonClassifier
from dataset import DataLoader, PokeDataset, read_classes
from matplotlib import pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def load_data(batch_size=100):
    test_dataloader = DataLoader(batch_size=batch_size, shuffle=False, max_size=80).get_data_loader()
    data = iter(test_dataloader).next()
    return data

if __name__=='__main__':
    classes = read_classes()
    data = load_data(64)

    images, labels, class_names = data
    print(images[0].shape[:])

    #for i in range(0,len(images)):
    grid = utils.make_grid(images)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.show()
    cv2.waitKey(0)