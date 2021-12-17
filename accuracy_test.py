import torch
import csv
import numpy
import cv2
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from utils import accuracy
from model import PokemonClassifier
from dataset import DataLoader, PokeDataset, read_classes


CSV_FILE = "pokedex.csv"
DATASET_DIR = "Pokemon_Images"
IMG_EXT = ".png"

POKEMONDATA = "PokemonData"
CLASSES_CSV = "classes.csv"


def load_model(path_to_model):
    model = PokemonClassifier()
    model.eval()
    PokemonClassifier.load_from_checkpoint(path_to_model)
    return model

def load_data(batch_size=1500):
    test_dataloader = DataLoader(batch_size=batch_size, shuffle=False, max_size=8000).get_data_loader()
    data = iter(test_dataloader).next()
    return data

if __name__=='__main__':
    path_to_model = "models/first_test.ckpt"
    model = load_model(path_to_model)
    classes = read_classes()
    data = load_data(63)
    images, labels, class_names = data
    pred_labels = model(images)
    class_propabilities, pred_labels = torch.max(pred_labels, 1)
    right_preds = 0

    for i in range(0,len(pred_labels)):
        print(i)
        print(pred_labels[i])
        print(labels[i])
        if pred_labels[i] == labels[i]:
            right_preds+=1


    print(right_preds)
    print(accuracy(labels, pred_labels))


