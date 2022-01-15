import os
#import cv2
import csv
import shutil

SOURCEFOLDER = "PokemonData"
TEACHERFOLDER = "PokemonDataTeacher"
STUDENTFOLDER = "PokemonDataStudent"

def preperFolders():

    shutil.copytree(SOURCEFOLDER, TEACHERFOLDER)
    shutil.copytree(SOURCEFOLDER, STUDENTFOLDER)

    directory = os.fsencode(STUDENTFOLDER)
    
    for folder in os.listdir(directory):
        #foldername = os.fsdecode(folder)[2:-1]
        subDirectory = os.fsencode(STUDENTFOLDER +"\\" + folder)
        iterator = 0
        for file in os.listdir(subDirectory):
            fileString = str(file)[2:]
            fileString = fileString[:-1]
            os.remove(STUDENTFOLDER +"\\" + foldername + "\\" + fileString)
            iterator = iterator + 1
            if iterator == 5:
                break
        for file in os.listdir(subDirectory):
            fileString = str(file)[2:]
            fileString = fileString[:-1]
            os.remove(TEACHERFOLDER +"\\" + foldername + "\\" + fileString)
            


