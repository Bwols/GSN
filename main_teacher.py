import hydra
from omegaconf import DictConfig, OmegaConf
from trainModel import train_model
import os
from hydra.utils import get_original_cwd
from data_preprocess import prepare_dataset
import torch
from preperSets import preperFolders
from label_student import label_student
CONFIG_PATH = "config"
CONFIG_NAME = "teacher_config"
DEFAULT_MODEL_NAME = "run_0"
TEACHER_MODEL_DIR = "teacher_models"
STUDENT_MODEL_DIR = "student_models"

STUDENTFOLDER = "Pokemon_Images_Student64"
TEACHERFOLDER= "Pokemon_Images_Teacher64"
CLASSES_CSV = "classes.csv"
CSV_FILE_STUDENT = "pokedex_student64.csv"
CSV_FILE_TEACHER = "pokedex_teacher64.csv"


#git add foldername/\*

teacher_student = False

@hydra.main(config_path=CONFIG_PATH,config_name=CONFIG_NAME)



def run_hydra(config):
    print(get_original_cwd(),"\n")
    os.chdir(get_original_cwd()) # <--- molto importanto

    print(OmegaConf.to_yaml(config))

    dataset_dir_teacher = config.dataset_config.dir_teacher
    dataset_dir_student = config.dataset_config.dir_student

    dataset_pokedex_teacher = config.dataset_config.pokedex_teacher
    dataset_pokedex_student = config.dataset_config.pokedex_student

    val_dataset_dir_teacher = config.dataset_config.val_dir_teacher
    val_dataset_dir_student = config.dataset_config.val_dir_student

    val_dataset_pokedex_teacher = config.dataset_config.val_pokedex_teacher
    val_dataset_pokedex_student = config.dataset_config.val_pokedex_student

    batch_size = int(config.training_config.batch_size)
    lr = float(config.training_config.lr)
    epochs = int(config.training_config.epochs)
    optimizer = config.training_config.optimizer
    loss_function = config.training_config.loss_function
    
    if not os.path.isdir(TEACHER_MODEL_DIR):
        os.mkdir(TEACHER_MODEL_DIR)
    if not os.path.isdir(STUDENT_MODEL_DIR):
        os.mkdir(STUDENT_MODEL_DIR)
        
    teacher_model_dir = TEACHER_MODEL_DIR
    student_model_dir = STUDENT_MODEL_DIR

    #device = config.training_config.device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = 'cuda'
    print("--> using:{}".format(device))

    architecture = config.architecture_config.architecture
    image_size = config.architecture_config.image_size


    try:
        model_name = config.name
    except:
        model_name = DEFAULT_MODEL_NAME

    
    # Musimy najpierw przygotowac folder "PokemonDataTeacher" i "PokemonDataStudent"
    preperFolders()
    # Teraz musimy zamienić te foldery na foldery "Pokemon_Images_Student" i "Pokemon_Images_Teacher"
    prepare_dataset(dim=(64,64), pokemonDataDir="PokemonDataTeacher", output_dir = TEACHERFOLDER, csv_file =CSV_FILE_TEACHER) # Zestaw danych o rozmiarze 64x64
    prepare_dataset(dim=(64,64), pokemonDataDir="PokemonDataStudent", output_dir = STUDENTFOLDER, csv_file =CSV_FILE_STUDENT) # Zestaw danych o rozmiarze 64x64
    # Nie potrzebny jest csv dla studenta
    os.remove(CSV_FILE_STUDENT) 

    # Teraz trenujemy model nauczyciela, ktory trafi do folderu teacher_models
    train_model(model_name=model_name,
            dataset_dir=dataset_dir_teacher,
            dataset_pokedex=dataset_pokedex_teacher,
            val_dataset_dir=val_dataset_dir_teacher,
            val_dataset_pokedex=val_dataset_pokedex_teacher,
            architecture=architecture,
            image_size=image_size,
            batch_size=batch_size,
            max_epochs=epochs,
            lr=lr,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            model_dir = teacher_model_dir
            )
    # Po przeprowadzeniu nauki w folderze teacher_models pojawia sie model nauczyciela
    # Teraz beda etykietowane nieuzyte dane przy uzyciu tego modelu 
    label_student(student_folder = STUDENTFOLDER , teacher_folder = TEACHERFOLDER, classes_csv = CLASSES_CSV , csv_file_student = CSV_FILE_STUDENT , csv_file_teacher = CSV_FILE_TEACHER, model_path =TEACHER_MODEL_DIR)

    # Teraz trzeba przeprowadzic nauke w oparciu o nowowytworzony zbior studenta
    train_model(model_name=model_name,
            dataset_dir=dataset_dir_student,
            dataset_pokedex=dataset_pokedex_student,
            val_dataset_dir=val_dataset_dir_student,
            val_dataset_pokedex=val_dataset_pokedex_student,
            architecture=architecture,
            image_size=image_size,
            batch_size=batch_size,
            max_epochs=epochs,
            lr=lr,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            model_dir = student_model_dir
            )

#print("Current Working Directory ", os.getcwd())
run_hydra()
