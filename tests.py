"""
Kod oparty o :
https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5


"""


import torch
from model import PokemonClassifier
from dataset import DataLoader, read_classes
import numpy as np
import matplotlib.pyplot as plt



from utils import accuracy
from torchmetrics import Accuracy

#model  = PokemonClassifier()
#PokemonClassifier.load_from_checkpoint("C:\\Users\\JeLoń\\Desktop\\GSN\\lightning_logs\\version_49\\checkpoints\\epoch=0-step=427.ckpt")


def load_model(path_to_model): # TODO
    model = PokemonClassifier()
    model.eval()
    model = model.load_from_checkpoint(path_to_model)
    return model

def load_data(batch_size=8,max_size=1000000):
    test_dataloader = DataLoader(batch_size=batch_size, shuffle=True, max_size=max_size).get_data_loader()
    data = iter(test_dataloader).next()
    return data


def show_results_of_model(model, data, output_file=None):
    images, labels, class_names = data
    fig = plt.figure(figsize=(15, 3))
    pred_labels = model(images)
    print(pred_labels[0])
    classes = read_classes()
    class_propabilities, pred_labels = torch.max(pred_labels, 1)

    right_preds = 0

    for i in range(0,len(pred_labels)):
        print(i)
        if pred_labels[i] == labels[i]:
            right_preds+=1


    print(right_preds)


    print(labels)
    print(pred_labels)
    print(accuracy(labels, pred_labels))

    for i in range(len(images)):
        image = images[i]
        image = np.transpose(image, (1, 2, 0))

        sub = fig.add_subplot(1, len(images), i + 1)
        true_class = class_names[i]
        pred_class = classes[pred_labels[i]]
        sub.set_title("Data: {} \n Pred: {} ".format(true_class, pred_class))
        plt.axis('off')
        plt.imshow(image)
    if output_file:
        plt.savefig(output_file)
    plt.show()


def calc_accuracy(mode_name, device="cuda"):
    test_dataloader = DataLoader(batch_size=16, shuffle=True, max_size=100000).get_data_loader()
    model = load_model(mode_name).to(device)

    accuracy = Accuracy().to(device)

    labels_arr = torch.tensor([]).to(device)
    pred_labels_arr = torch.tensor([]).to(device)

    for i, data in enumerate(test_dataloader):
        images, labels, names = data
        labels = labels.to(device)
        labels_arr = torch.cat((labels_arr, labels), 0)

        pred_labels = model(images.to(device))
        class_propabilities, pred_labels = torch.max(pred_labels, 1)
        pred_labels_arr = torch.cat((pred_labels_arr, pred_labels), 0)

    print(labels_arr.int())
    print(pred_labels_arr.int())

    labels_arr = labels_arr.to(device)
    pred_labels_arr = pred_labels_arr.to(device)

    acc = (labels_arr.int(), pred_labels_arr.int())
    print("Accuracy:", acc.float())
    return accuracy

