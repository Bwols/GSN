import os
import cv2
import csv
from glob import glob

DATASET_DIR = "Pokemon_Images"
POKEMONDATA = "PokemonData"
CSV_FILE = "pokedex.csv"




def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def crop(image,dim=(64,64)):
    new_img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return new_img


def image_transformation(image_path):
    image = cv2.imread(image_path)
    return crop(image)


def prepare_dataset(pokemonDataDir=POKEMONDATA ,output_dir = DATASET_DIR):

    i = 0  # numer obrazu
    k = 0  # numer klasy

    make_dir(output_dir)

    file = open(CSV_FILE, 'w', newline='')
    writer = csv.writer(file)
    #writer.writerow(["image number", "pokemon name", "pokemon class index"])

    for pokemon_folder in os.listdir(pokemonDataDir):
        pokemon_name = pokemon_folder
        pokemon_folder = os.path.join(pokemonDataDir,pokemon_folder)

        if os.path.isdir(pokemon_folder):
            #print(pokemon_folder)
            k += 1
            for filename in os.listdir(pokemon_folder):

                image_path = os.path.join(pokemon_folder, filename)

                #print(pokemon_name,filename,image.shape[:])
                if not filename.endswith(".jpg") and not filename.endswith(".png") and not filename.endswith(".jpeg"):
                    print("         ERROR READING FILE",pokemon_name,filename)

                print(pokemon_name, filename)

                i += 1
                new_image = image_transformation(image_path)
                image_name = i
                out_file = "{}/{}.png".format(output_dir, image_name)
                cv2.imwrite(out_file, new_image)
                writer.writerow([i,pokemon_name,k])

    file.close()


prepare_dataset()


