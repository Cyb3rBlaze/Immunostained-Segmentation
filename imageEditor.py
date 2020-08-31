from PIL import Image
from os import listdir
from tqdm import tqdm
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def save_training_data(directory):
    images = []
    size = (3500, 4500)
    image = Image.open("./raw.jpg").convert("RGB").resize(size)
    count = 0
    for i in tqdm(range(7)):
        for j in range(9):
            temp = image.crop((i*500, j*500, i*500 + 500, j*500 + 500))
            temp.save(directory + "/" + str(count) + ".jpg")
            count += 1

def create_labels(directory):
    images = listdir(directory)
    npImages = []
    for i in images:
        image_ = Image.open(directory + "/" + i).convert("RGB").resize((500, 500))
        npImages += [np.array(image_)]

    print("Filter phase 1...")

    index = 0
    for npImage in tqdm(npImages):
        for i in range(len(npImage)):
            for j in range(len(npImage[i])):
                if npImage[i][j][0] < 220 or npImage[i][j][0] > 260:
                    npImages[index][i][j] = [0, 0, 0]
        index += 1
    
    print("Filter phase 2...")

    index = 0
    for npImage in tqdm(npImages):
        for i in range(len(npImage)):
            for j in range(len(npImage[i])):
                if npImage[i][j][0] > 170 and npImage[i][j][1] > 170 and npImage[i][j][2] > 170:
                    npImages[index][i][j] = [255, 255, 255]
                elif not(npImage[i][j][0] == 0 and npImage[i][j][1] == 0 and npImage[i][j][2] == 0):
                    npImages[index][i][j] = [255, 90, 100]
        index += 1

    
    print("Filter phase 3...")
    for npImage in range(len(npImages)):
        im_rgb = cv2.cvtColor(npImages[npImage], cv2.COLOR_BGR2RGB)
        cv2.imwrite("raw_labels/" + images[npImage], im_rgb)

def produceAugmentations(data, masks):
    images = listdir(data)
    cv2Images = []
    cv2Masks = []
    for i in images:
        cv2Images += [cv2.imread(data + "/" + i)]
        cv2Images += [cv2.imread(masks + "/" + i)]

produceAugmentations()