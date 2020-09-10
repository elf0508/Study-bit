# 불러와서 모델을 완성하시오.

import numpy as np

import pandas as pd

datasets = np.load('./data/data_sam_np_save.npy') 
datasets = np.load('./data/data_hit_np_save.npy') 

print(datasets.shape)  

x = datasets[:, :4]

print(x.shape)       

y = datasets[:, 4:]

print(y.shape)      
  