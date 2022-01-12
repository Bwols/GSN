"""
Kod oparty o https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09

"""
import pytorch_lightning as pl
from model import PokemonClassifier

from dataset import DataLoader
from data_preprocess import make_dir
import time
from tests import show_results_of_model, load_data

MODELS_DIR = "models"
CUDA = 'cuda'

def train_model(model_name, dataset_dir, dataset_pokedex, val_dataset_dir,val_dataset_pokedex, architecture="ResNet50",image_size=60,
                batch_size=16, max_epochs=5,lr=0.001,optimizer="Adam",loss_function="CrossEntropy",device="cpu"):
    make_dir(MODELS_DIR)
    model = PokemonClassifier(architecture=architecture, image_size=image_size, lr=lr, optimizer=optimizer, loss_function=loss_function,
                              device=device)

    train_dataloader = DataLoader(dataset_dir=dataset_dir,
                                  labels_csv=dataset_pokedex,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  max_size=10).get_data_loader()

    val_dataloader = DataLoader(dataset_dir=val_dataset_dir,
                                  labels_csv=val_dataset_pokedex,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  max_size=10).get_data_loader()

    gpus = 0
    if device == CUDA:
        gpus = 1

    trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus)

    a = time.time()
    trainer.fit(model, train_dataloader,val_dataloader)#, val_loader
    b = time.time()
    print("\nCzas trenowania: ",int(b - a), "s")
    make_dir(MODELS_DIR)

    model_save_path = "{}/{}.ckpt".format(MODELS_DIR, model_name)
    trainer.save_checkpoint(model_save_path)
    print("model zapisany w:", model_save_path)

    data = load_data(16)
    show_results_of_model(model, data, output_file="in_train.png")




