import torch
print(torch.__version__)
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

#read_csv_file()

class PokeDataset(Dataset):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0,), (1,))  # zakres 0,1
         ])


    def __init__(self, dataset_dir=DATASET_DIR, labels_csv=CSV_FILE):

        self.data = []
        csv_labels = read_csv_file(labels_csv)
        csv_labels = np.transpose(csv_labels)
        #print(csv_labels)

        for image_name in os.listdir(dataset_dir):

            num = image_name.replace(IMG_EXT,'')
            label_idx = np.where(csv_labels[0] ==num)


            image_path = os.path.join(dataset_dir,image_name)
            image = cv2.imread(image_path)
            dat = (image, int(csv_labels[2][label_idx][0]), csv_labels[1][label_idx][0])
            self.data.append(dat)




    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class DataLoader:

    def __init__(self, batch_size=16, shuffle=False):

        dataset = PokeDataset()
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False
        )

    def get_data_loader(self):
        return self.data_loader

data_loader =  DataLoader().get_data_loader()

for i, data in enumerate(data_loader):
    print(i)
    print(len(data[0]))
