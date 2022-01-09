"""
Do stworzenie tego programu został wykorzystany kod z https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09
i https://towardsdatascience.com/keeping-up-with-pytorch-lightning-and-hydra-2nd-edition-34f88e9d5c90
"""

import pytorch_lightning as pl
import torch
from resnet import ResNet
from torchmetrics import Accuracy


def ResNet50(inputSize=64):
    return ResNet([3, 4, 6, 3], inputSize=inputSize)

def ResNet152(inputSize=64):
    return ResNet([3, 8, 36, 3], inputSize=inputSize)



class PokemonClassifier(pl.LightningModule):

    def __init__(self, architecture="ResNet50", image_size=64, lr=0.001):
        super(PokemonClassifier, self).__init__()

        # self.model = Model()
        self.lr = lr

        self.model = None
        if architecture == "ResNet50":
            self.model = ResNet50(inputSize=image_size)
        elif architecture =="ResNet152":
            self.model = ResNet152(inputSize=image_size)

        else:
            print("Unknown Architecture!!!")


    def cross_entropy_loss(self, logits, labels):

        loss = torch.nn.CrossEntropyLoss(reduction='mean')
        return loss(logits, labels)

    def forward(self, x):

        x = self.model(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, labels, names = train_batch

        logits = self.forward(x)

        loss = self.cross_entropy_loss(logits, labels)
        self.log('train_loss', loss)
        return loss



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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)#lr=1e-3
        #optimizer = torch.optim.SGD(self.parameters(), lr=0.001)  # lr=1e-3
        return optimizer

