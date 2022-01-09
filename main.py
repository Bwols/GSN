import hydra
from omegaconf import DictConfig, OmegaConf
from trainModel import train_model
CONFIG_PATH = "config"
CONFIG_NAME = "default_config"
DEFAULT_MODEL_NAME = "run_0"
from os import listdir
import os
from hydra.utils import get_original_cwd


def open_pokedex():
    print(listdir("."))

@hydra.main(config_path=CONFIG_PATH,config_name=CONFIG_NAME)

def run_hydra(config):

    print(get_original_cwd())
    os.chdir(get_original_cwd()) # <--- molto importanto

    print(OmegaConf.to_yaml(config))

    dataset_dir = config.dataset_config.dir
    dataset_pokedex = config.dataset_config.pokedex

    batch_size = int(config.training_config.batch_size)
    lr = float(config.training_config.lr)
    epochs = int(config.training_config.epochs)

    architecture = config.architecture_config.architecture
    image_size = config.architecture_config.image_size

    try:
        model_name = config.name
    except:
        model_name = DEFAULT_MODEL_NAME



    train_model(model_name=model_name,
                dataset_dir=dataset_dir,
                dataset_pokedex=dataset_pokedex,
                architecture=architecture,
                image_size=image_size,
                batch_size=batch_size,
                max_epochs=epochs,
                lr=lr
                )




#print("Current Working Directory ", os.getcwd())
run_hydra()
