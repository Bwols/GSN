import torch
#print(torch.__version__)
import csv
import numpy
import cv2
#import pytorch_lightning as pl
#1.9.0+cu111
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np



CSV_FILE = "pokedex.csv"
DATASET_DIR = "Pokemon_Images"
IMG_EXT = ".png"

POKEMONDATA = "PokemonData"
CLASSES_CSV = "classes.csv"


def read_csv_file(csv_file_name = CSV_FILE):
    file = open(csv_file_name)
    csvreader = csv.reader(file)
    #header = next(csvreader)
    #print(header)
    rows = []
    for row in csvreader:
        rows.append(row)

    #print(rows)
    file.close()
    return rows


class PokeDataset(Dataset):

    def __init__(self, dataset_dir=DATASET_DIR, labels_csv=CSV_FILE, max_size=10000):
        j = 0

        self.data = []
        csv_labels = read_csv_file(labels_csv)
        csv_labels = np.transpose(csv_labels)
        #print(csv_labels)

        oirignalTransform = transforms.Compose(
            [
             transforms.ToPILImage(),
             transforms.ToTensor(),
             transforms.Normalize((0,), (1,)),  # zakres 0,1
             ])

        transform = transforms.Compose(
            [
             transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(), # Augemntacja
             transforms.RandomVerticalFlip(), # Augmentacja
             transforms.RandomRotation(degrees = 45), # Augmentacja 
             transforms.ColorJitter(brightness=0, contrast = 0, saturation = 0), # Augmentacja
             transforms.RandomGrayscale(), # Augemntacja
             transforms.ToTensor(),
             transforms.Normalize((0,), (1,)),  # zakres 0,1
             ])

        for image_name in os.listdir(dataset_dir):

            num = image_name.replace(IMG_EXT,'')
            label_idx = np.where(csv_labels[0] == num)


            image_path = os.path.join(dataset_dir,image_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#TODO wczytwać wszędzie w tym standardzie
            imageOriginal = oirignalTransform(image)
            dat = (imageOriginal, int(csv_labels[2][label_idx][0]), csv_labels[1][label_idx][0]) #zwraca obraz, label, i nazwę klasy #TODO Usunięte
            self.data.append(dat) 
            #print(image.shape[:])
            #dat = (image2, int(csv_labels[2][label_idx][0]), csv_labels[1][label_idx][0]) #zwraca obraz, label, i nazwę klasy #TODO Usunięte
            for i in range(3):
                image2 = transform(image)
                dat = (image2, int(csv_labels[2][label_idx][0]), csv_labels[1][label_idx][0]) #zwraca obraz, label, i nazwę klasy #TODO Usunięte
                self.data.append(dat)

            j+=1
            if j >= max_size:
                return


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class DataLoader:

    def __init__(self, dataset_dir=DATASET_DIR, labels_csv=CSV_FILE, batch_size=16, shuffle=False,max_size=100000):
        print("Loaded: ",dataset_dir)
        dataset = PokeDataset(dataset_dir, labels_csv, max_size)
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False
        )

    def get_data_loader(self):
        return self.data_loader


def read_classes(classes_csv=CLASSES_CSV):

    csv_labels = read_csv_file(classes_csv)

    data = [i for i in range(150)]
    for t in csv_labels:
        data[int(t[0])] = t[1]
    return data