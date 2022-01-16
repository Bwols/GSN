import os
#import cv2
import csv
import shutil

SOURCEFOLDER = "PokemonData"
TEACHERFOLDER = "PokemonDataTeacher"
STUDENTFOLDER = "PokemonDataStudent"

def preperFolders():

    if not os.path.isdir(TEACHERFOLDER):
        shutil.copytree(SOURCEFOLDER, TEACHERFOLDER)
    else:
      return
    if not os.path.isdir(STUDENTFOLDER):   
        shutil.copytree(SOURCEFOLDER, STUDENTFOLDER)
    else:
      return

    directory = STUDENTFOLDER
    
    for folder in os.listdir(directory):
        #foldername = os.fsdecode(folder)[2:-1]
        subDirectory = STUDENTFOLDER +"/" + str(folder)
        iterator = 0
        for file in os.listdir(subDirectory):
            fileString = str(file)
            os.remove(STUDENTFOLDER +"/" + str(folder) + "/" + fileString)
            iterator = iterator + 1
            if iterator == 5:
                break
        for file in os.listdir(subDirectory):
            fileString = str(file)
            os.remove(TEACHERFOLDER +"/" + str(folder) + "/" + fileString)
