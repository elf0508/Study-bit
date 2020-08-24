## 필요한 패키지 등록
# https://bizzengine.tistory.com/188

import os
import cv2
import numpy as np
from sklearn.utils import shuffle

from PIL import Image
import matplotlib.pyplot as plt

Images = []
Labels = []

directory = 'T_project/Low_Resolution'

for label, names in enumerate(os.listdir(directory)):
    try:
        for image_file in os.listdir(directory + names):
            image = cv2.imread(directory + r'/' + image_file)
            image = cv2.resize(image, (150, 150))
            image.append(image)
            Labels.append(label)

    except Exception as e:
        print(str(e))

shuffle(Images, Labels, random_state = 5)

Images = np.array(Images)
Labels = np.array(Labels)

file_names = 'alien_predator_train'

Save = np.savez(directory + file_names, x = Images, y = Labels)

print(Images.shape)
print(Labels.shape)





