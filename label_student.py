import os
import torch
from tests import show_results_of_model, load_data, load_model, calc_accuracy
from PIL import Image
import torchvision
import torchvision.transforms as transforms
imsize = 256
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
import cv2 as cv


transform = transforms.Compose(
            [
             transforms.ToTensor(),
             transforms.Normalize((0,), (1,)),  # zakres 0,1
             ])

def label_student(path_to_model):
    #directory = os.fsencode("models")
    #path_to_model = str(directory)[2:-1] + "\\" +  str(os.listdir(directory))[3:-2]

    model = load_model(path_to_model,architecture="ResNet50",image_size=64)
    model = model.cuda()
    model.eval()

    image = image_loader("Pokemon_Images/1.png")
    print(image.shape[:])
    x = model(image)
    print(x)


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = transform(image).to('cuda').unsqueeze(0)
    return image

label_student("models/run_0_ResNet50.ckpt")
