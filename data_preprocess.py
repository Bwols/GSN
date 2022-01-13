import os
import cv2
import csv
from glob import glob

DATASET_DIR = "Pokemon_Images_Teacher"
POKEMONDATA = "PokemonDataTeacher"
CSV_FILE = "pokedex.csv"
CLASSES_CSV = "classesteacher.csv"



def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def crop(image,dim=(64,64)):#change here size
    new_img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return new_img


def image_transformation(image_path,dim=(64,64)):
    image = cv2.imread(image_path)
    return crop(image,dim)


def prepare_dataset(dim, pokemonDataDir=POKEMONDATA, output_dir = DATASET_DIR, csv_file =CSV_FILE):

    i = 0  # numer obrazu
    k = -1  # numer klasy

    make_dir(output_dir)

    file = open(csv_file, 'w', newline='')
    writer = csv.writer(file)

    file2 = open(CLASSES_CSV, 'w', newline='')
    writer2 = csv.writer(file2)
    #writer.writerow(["image number", "pokemon name", "pokemon class index"])

    for pokemon_folder in os.listdir(pokemonDataDir):
        pokemon_name = pokemon_folder
        pokemon_folder = os.path.join(pokemonDataDir,pokemon_folder)

        if os.path.isdir(pokemon_folder):
            #print(pokemon_folder)
            k += 1
            writer2.writerow([k, pokemon_name])
            for filename in os.listdir(pokemon_folder):

                image_path = os.path.join(pokemon_folder, filename)

                #print(pokemon_name,filename,image.shape[:])
                if not filename.endswith(".jpg") and not filename.endswith(".png") and not filename.endswith(".jpeg"):
                    print("         ERROR READING FILE",pokemon_name,filename)

                print(pokemon_name, filename)

                i += 1
                new_image = image_transformation(image_path,dim)
                image_name = i

                out_file = "{}/{}.png".format(output_dir, image_name)
                cv2.imwrite(out_file, new_image)
                writer.writerow([i,pokemon_name,k])


    file.close()
    file2.close()

#uncomment to run prepare dataset
#prepare_dataset(dim=(256, 256), output_dir="Pokemon_Images_256",csv_file="pokedex_256.csv")


