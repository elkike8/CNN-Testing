import numpy as np
import random
import os, shutil
import matplotlib.pyplot as plt

filepath = "C:/Users/FerGo/OneDrive/ACIT/01.2022/Data Mining/Project/Images"
list_of_images = os.listdir(filepath)
random.seed(42)

total_images = 112120 - 1
num_images = int(np.round(total_images * 0.01))

indexes = []
images = 0

while images < num_images-1:
    a = random.randrange(total_images)
    if a in indexes:
        print("ignoring " + str(a))
    else:
        try:
            image = plt.imread(filepath + "/" + list_of_images[a])
            image.shape[2]
            print("4d image " + str(a))
        except IndexError:
            indexes.append(a)
            images += 1
            
            



random_images = []

for i in indexes:
    random_images.append(list_of_images[i])
    
destiny = "C:/Users/FerGo/OneDrive/ACIT/01.2022/Data Mining/Project/Selected X_rays"

for i in random_images:
    shutil.copyfile(filepath + "/" + i,
                    destiny + "/" + i)