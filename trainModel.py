"""
Kod oparty o https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09

"""



import pytorch_lightning as pl
from model import PokemonClassifier

from dataset import DataLoader
from data_preprocess import make_dir

from tests import show_results_of_model, load_data

MODELS_DIR = "models"


def train_model(model_name, dataset_dir, dataset_pokedex,architecture="ResNet50",image_size=60, batch_size=16, max_epochs=5,
                lr=0.001,optimizer="Adam",loss_function="CrossEntropy"):
    make_dir(MODELS_DIR)
    model = PokemonClassifier(architecture=architecture, image_size=image_size, lr=lr, optimizer=optimizer, loss_function=loss_function)

    train_dataloader = DataLoader(dataset_dir=dataset_dir,
                                  labels_csv=dataset_pokedex,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  max_size=10).get_data_loader()

    #val_loader = DataLoader(batch_size=64,shuffle=True).get_data_loader()

    trainer = pl.Trainer(max_epochs=max_epochs,gpus=1)

    trainer.fit(model, train_dataloader)#, val_loader

    make_dir(MODELS_DIR)
    trainer.save_checkpoint("{}/{}.ckpt".format(MODELS_DIR, model_name))

    data = load_data(16)
    show_results_of_model(model, data, output_file="in_train.png")




