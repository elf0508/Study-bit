import urllib.request
# from bs4 import BeautifulSoup
import urllib.error
import urllib.parse
import urllib.robotparser
from urllib.request import Request, urlopen
# import re
import requests
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
from urllib.parse import quote_plus
import PIL.Image as pilimg
import cv2

a = plt.imread('C:/Users/bitcamp/Desktop/img/puppy-picture-2.jpg')
plt.imshow

a = pilimg.open('C:/Users/bitcamp/Desktop/img/puppy-picture-2.jpg')
ap = np.array(a)

a = cv2.imread('C:/Users/bitcamp/Desktop/img/puppy-picture-2.jpg')
cv2.imshow('title', a)
cv2.waitKey()
















# headers = {'User-Agent': 'Mozilla/5.0'}
# baseUrl = 'https://www.google.co.kr/search?q=puppy++with+eyes+open&tbm=isch&ved=2ahUKEwidiLvcxYLqAhUH0ZQKHcbcBkAQ2-cCegQIABAA&oq=puppy++with+eyes+open&gs_lcp=CgNpbWcQAzIGCAAQBxAeMgYIABAHEB4yBggAEAgQHjoICAAQCBAHEB5QowpYjxJgjxVoAHAAeACAAW-IAdcEkgEDNC4ymAEAoAEBqgELZ3dzLXdpei1pbWc&sclient=img&ei=ML7mXp2wG4ei0wTGuZuABA&bih=937&biw=1920&hl=ko#imgrc=zkRF1ZC3Uz5JGM'
# plusUrl = input('검색어 입력 : ')
# crawl_num = int(input('크롤링할 갯수 입력(최대 50개) : '))

# url = baseUrl + quote_plus(plusUrl) # 한글 검색 자동 변환
# html = urlopen(url)
# soup = bs(html, "html.parser")
# img = soup.find_all(class_='_img')
 
# n = 1
# for i in img:
#     print(n)
#     imgUrl = i['data-source']
#     with urlopen(imgUrl) as f:
#         with open('./images/img' + str(n)+'.jpg','wb') as h: # w - write b - binary
#             img = f.read()
#             h.write(img)
#     n += 1
#     if n > crawl_num:
#         break
    
    
# print('Image Crawling is done.')








'''
url="https://www.google.co.kr/search?q=puppy++with+eyes+open&tbm=isch&ved=2ahUKEwidiLvcxYLqAhUH0ZQKHcbcBkAQ2-cCegQIABAA&oq=puppy++with+eyes+open&gs_lcp=CgNpbWcQAzIGCAAQBxAeMgYIABAHEB4yBggAEAgQHjoICAAQCBAHEB5QowpYjxJgjxVoAHAAeACAAW-IAdcEkgEDNC4ymAEAoAEBqgELZ3dzLXdpei1pbWc&sclient=img&ei=ML7mXp2wG4ei0wTGuZuABA&bih=937&biw=1920&hl=ko#imgrc=zkRF1ZC3Uz5JGM"
site = requests.get(url)
soup = BeautifulSoup(site.content,"html.parser")

table = soup.find("img")['src']

# 시각화

plt.show()
'''






