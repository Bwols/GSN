import torch
from torchmetrics import Accuracy



def accuracy(labels,pred_labels):
    accuracy = Accuracy()
    acc = accuracy(labels, pred_labels)
    return acc