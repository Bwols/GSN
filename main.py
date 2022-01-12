import hydra
from omegaconf import DictConfig, OmegaConf
from trainModel import train_model
import os
from hydra.utils import get_original_cwd
import torch
CONFIG_PATH = "config"
CONFIG_NAME = "default_config"
DEFAULT_MODEL_NAME = "run_0"

#git add foldername/\*

@hydra.main(config_path=CONFIG_PATH,config_name=CONFIG_NAME)

def run_hydra(config):

    print(get_original_cwd(),"\n")
    os.chdir(get_original_cwd()) # <--- molto importanto

    print(OmegaConf.to_yaml(config))

    dataset_dir = config.dataset_config.dir
    dataset_pokedex = config.dataset_config.pokedex
    val_dataset_dir = config.dataset_config.val_dir
    val_dataset_pokedex = config.dataset_config.val_pokedex

    batch_size = int(config.training_config.batch_size)
    lr = float(config.training_config.lr)
    epochs = int(config.training_config.epochs)
    optimizer = config.training_config.optimizer
    loss_function = config.training_config.loss_function

    #device = config.training_config.device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    print("--> using:{}".format(device))

    architecture = config.architecture_config.architecture
    image_size = config.architecture_config.image_size


    try:
        model_name = config.name
    except:
        model_name = DEFAULT_MODEL_NAME


    train_model(model_name=model_name,
                dataset_dir=dataset_dir,
                dataset_pokedex=dataset_pokedex,
                val_dataset_dir=val_dataset_dir,
                val_dataset_pokedex=val_dataset_pokedex,
                architecture=architecture,
                image_size=image_size,
                batch_size=batch_size,
                max_epochs=epochs,
                lr=lr,
                optimizer=optimizer,
                loss_function=loss_function,
                device=device
                )




#print("Current Working Directory ", os.getcwd())
run_hydra()
