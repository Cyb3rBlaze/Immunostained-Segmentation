import os
from tqdm import tqdm
import glob
from PIL import Image

def convertToJPG(directory):
    current_dir = os.getcwd()
    pokemon_dir = os.path.join(current_dir, directory)
    images = []
    for filename in tqdm(glob.glob(directory+'/*.tif')):
        image = Image.open(filename).convert("RGB")
        image.save("new/" + filename[9:len(filename)-3] + "jpg")

convertToJPG("./images")