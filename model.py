"""
Do stworzenie tego programu został wykorzystany kod z https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09

"""

import pytorch_lightning as pl
import torch
from resnet import ResNet
from torchmetrics import Accuracy


def ResNet50():
    return ResNet([3, 4, 6, 3], inputSize=256)

def ResNet152():
    return ResNet([3, 8, 36, 3], inputSize=64)



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

