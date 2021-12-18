import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from resnet import ResNet
#from utils import accuracy
from torchmetrics import Accuracy
import torch.nn as nn

def ResNet50():
    return ResNet([3, 4, 6, 3], inputSize=64)

def ResNet152():
    return ResNet([3, 8, 36, 3], inputSize=64)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 150)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        #fun = nn.Sigmoid()
        #x = fun(x)
        return x



class PokemonClassifier(pl.LightningModule):

    def __init__(self):
        super(PokemonClassifier, self).__init__()

        # self.model = Model()
        self.model = ResNet50()

    def cross_entropy_loss(self, logits, labels):
        #loss = torch.nn.BCELoss(reduction='sum')
        loss = torch.nn.CrossEntropyLoss(reduction='mean')
        #loss = torch.nn.L1Loss(reduction='sum')
        # return F.nll_loss(logits, labels)

        return loss(logits, labels)

    def forward(self, x):
        #x.to("cuda")
        x = self.model(x)

        return x

    def training_step(self, train_batch, batch_idx):
        x, labels, names = train_batch

        #label = F.one_hot(label, 150)

        # print(x.size())
        logits = self.forward(x)
        #print(labels)
        loss = self.cross_entropy_loss(logits, labels)
        self.log('train_loss', loss)
        return loss

    """

    def training_step(self, train_batch, batch_idx):
        x, label, name = train_batch
        #label = F.one_hot(label, 150)

        # print(x.size())
        logits = self.forward(x)
        class_propabilities, pred_labels = torch.max(logits, 1)

        loss = self.cross_entropy_loss(pred_labels, label)

        self.log('train_loss', loss)
        return loss
         """

    def validation_step(self, val_batch, batch_idx):
        x, labels, name = val_batch
        #labels_cp = F.one_hot(labels, 150)
        logits = self.forward(x)

        class_propabilities, pred_labels = torch.max(logits, 1)
        right_preds = 0
        accuracy = Accuracy().to("cuda")
        print("Accuracy:{}", accuracy(labels, pred_labels))
        print("labels", labels)
        print("pred", pred_labels)
        loss = self.cross_entropy_loss(logits, labels)

        self.log('val_loss', loss)




    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)#lr=1e-3
        #optimizer = torch.optim.SGD(self.parameters(), lr=0.001)  # lr=1e-3
        return optimizer

