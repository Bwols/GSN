import os
import torch
from tests import show_results_of_model, load_data, load_model, calc_accuracy
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import csv
from pathlib import Path

imsize = 64
STUDENTFOLDER = "Pokemon_Images_Student64"
TEACHERFOLDER= "Pokemon_Images_Teacher64"
CLASSES_CSV = "classes.csv"
CSV_FILE_STUDENT = "pokedex_student64.csv"
CSV_FILE_TEACHER = "pokedex_teacher64.csv"
MODEL_PATH = "teacher_models"

loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

transform = transforms.Compose(
            [
             transforms.ToTensor(),
             transforms.Normalize((0,), (1,)),  # zakres 0,1
             ])

def label_student(student_folder = STUDENTFOLDER , teacher_folder = TEACHERFOLDER, classes_csv = CLASSES_CSV , csv_file_student = CSV_FILE_STUDENT , csv_file_teacher = CSV_FILE_TEACHER, model_path =MODEL_PATH)  :
    directory = os.fsencode("models")
    path_to_model = str(directory)[2:-1] + "\\" +  str(os.listdir(directory))[3:-2] 

    model = load_model(path_to_model, architecture="ResNet50", image_size=64)
    model = model.cuda()
    model.eval()

    create_pokedex(model)
    merge_datasets()
    
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = transform(image).to('cuda').unsqueeze(0)
    return image

def create_pokedex(model):
    path, dirs, files = next(os.walk(STUDENTFOLDER))
    pokedex = open(CSV_FILE_STUDENT, 'w', newline='')
    writer = csv.writer(pokedex)
    for i in range(len(files)):
        filename = str(i+1) + ".png"
        image  = image_loader(STUDENTFOLDER + "\\" + filename)
        class_nr =int(torch.argmax(model(image)).item())

        with open(CLASSES_CSV) as f:
            classes = list(csv.reader(f))
            #print(classes[CLASSES_CSV])
        pokemon_name = classes[class_nr][1]
        print(pokemon_name)
        
        writer.writerow([i + 1, pokemon_name, class_nr])

def merge_datasets():
    path, dirs, files = next(os.walk(STUDENTFOLDER))
    nr_of_files_student = len(files)
    path, dirs, files = next(os.walk(TEACHERFOLDER))
    nr_of_files_teacher = len(files)

    for i in range(nr_of_files_teacher): # Iterujemy po folderze teachera
        nr_of_files_student = nr_of_files_student + 1 
        os.replace(TEACHERFOLDER + "\\" + str(i+1) + ".png",STUDENTFOLDER + "\\" + str(nr_of_files_student) + ".png") # dodajemy rzeczy z folderu teachera na koniec folderu studenta
        print("Przeniosłem" + TEACHERFOLDER + "\\" + str(i+1) + ".png" + " do " +  STUDENTFOLDER + "\\" + str(nr_of_files_student) + ".png")
        pokedex = open(CSV_FILE_STUDENT, 'a', newline='') # Otwieramy utworzony wczesniej pokedex dla studenta
        writer = csv.writer(pokedex)
        with open(CSV_FILE_TEACHER) as f: 
            teacher_records = list(csv.reader(f))
        writer.writerow([nr_of_files_student, teacher_records[i][1], teacher_records[i][2]])