from PIL import Image
from tqdm import tqdm
import os

img_list = os.listdir("./images")
print(f"Number of images before filtering : {len(img_list)}")

for img in tqdm(img_list):
    img_path = "./images" + "/" + img
    try:
        
        pil_img = Image.open(img_path)
        
        w, h = pil_img.size
        pil_img.close()
        if w < 513 or h < 513:
            os.remove(img_path)
    except:
        os.remove(img_path)

img_list = os.listdir("./images")
print(f"Number of images after filtering : {len(img_list)}")
