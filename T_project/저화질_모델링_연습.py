
# 참고 : https://www.youtube.com/watch?v=3LNHxezPi1I

import dlib, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
import csv
def load_images_from_folder(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)  # 而щ윭�ъ쭊�� �댁슜�쒕떎.
    img = img[10:120, 20:150]  # �대�吏�瑜� �쇨뎬遺�遺꾨쭔 �섏삤寃� �섎씪以���.
    return img
def search(dirname):
    filenames = os.listdir(dirname)
    b=[]
    v=[]
    n=[]
    x=[]
    Image = []
    for i in filenames:
        if i == 'S001':
            x.append(os.path.join('{}'.format(dirname), i))
        else:
            continue
        for dirname2 in x:
            l = os.listdir('{}'.format(dirname2))
            for i in l:
                if i == 'L1':
                    b.append(os.path.join('{}'.format(dirname2), i))
                elif i == 'L2':
                    b.append(os.path.join('{}'.format(dirname2), i))
                # elif i == 'L3':
                #     b.append(os.path.join('{}'.format(dirname2), i))
                # elif i == 'L4':
                #     b.append(os.path.join('{}'.format(dirname2), i))
                else:
                    continue
                for dirname3 in b:
                    e = os.listdir('{}'.format(dirname3))
                    for i in e:
                        if i == 'E01':
                            v.append(os.path.join('{}'.format(dirname3), i))
                        elif i == 'E02':
                            v.append(os.path.join('{}'.format(dirname3), i))
                        else:
                            continue
                        for dirname4 in v:
                            c = os.listdir('{}'.format(dirname4))
                            for i in c:
                                if i == 'C7.jpg':
                                    n.append(os.path.join('{}'.format(dirname4), i))
                                # elif i == 'C6.jpg':
                                #     n.append(os.path.join('{}'.format(dirname4), i))
                                # elif i == 'C8.jpg':
                                #     n.append(os.path.join('{}'.format(dirname4), i))
                                else:
                                    continue
                                for i in tqdm(n):
                                    I = load_images_from_folder(i)

                                    Image.append(I)

    # f = np.array(Image)
    return Image
search('C:/Users/bitcamp/Downloads/Middle_Resolution/19062421')

#####
detector = dlib.get_frontal_face_detector() # �쇨뎬�먯�紐⑤뜽
sp = dlib.shape_predictor('teamproject/simple_face_recognition-master/models/shape_predictor_68_face_landmarks.dat') # �쒕뱶留덊겕 �먯� 紐⑤뜽
facerec = dlib.face_recognition_model_v1('teamproject/simple_face_recognition-master/models/dlib_face_recognition_resnet_model_v1.dat')#�쇨뎬�몄떇紐⑤뜽

def find_faces(img): # �쇨뎬 李얜뒗 �⑥닔
    dets= detector(img, 1)

    if len(dets) == 0: # �쇨뎬 紐살갼�쇰㈃ 鍮� 諛곗뿴 諛섑솚
        return np.empty(0), np.empty(0), np.empty(0)

    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int ) # 68媛쒖쓽 �쒕뱶留덊겕 援ы븯�� �⑥닔 / 0�� �됰젹濡� ndarray 諛섑솚�섎뒗 �⑥닔
    for k, d in enumerate(dets): # �쇨뎬留덈떎 猷⑦봽瑜� �뚯븘
        rect = ((d.left(), d.top()), (d.right(), d.bottom())) # �쇨뎬 �� �� �ㅻⅨ �꾨옒 醫뚰몴瑜� �ｌ뼱以���
        rects.append(rect)
        
        # landmark
        shape = sp(img, d) # �대�吏�, �ш컖�� �ｌ쑝硫� 68媛� �쒕뱶留덊겕 �섏�

    # convert dlib shape to numpy array
        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape) # �쒕뱶留덊겕 寃곌낵臾쇱쓣 �볦븘以�
    
    return rects, shapes, shapes_np

def encode_faces(img, shapes): # �쇨뎬�몄퐫�� -68媛� �쒕뱶留덊겕瑜� �몄퐫�붿뿉 �ｌ쑝硫� 128媛쒖쓽 踰≫꽣媛� �섏삤怨�, 洹멸구濡� �щ엺 �쇨뎬 援щ텇. 媛숈� �щ엺�몄� �꾨땶吏�
    face_descriptors=[]
    for shape in shapes: # �쒕뱶留덊겕 吏묓빀留뚰겮 猷⑦봽 �뚮━硫댁꽌 
        face_descriptor = facerec.compute_face_descriptor(img, shape) # �쇨뎬 �몄퐫�� (�꾩껜�대�吏�, 媛� �щ엺�ㅼ쓽 �쒕뱶留덊겕)
        face_descriptors.append(np.array(face_descriptor)) # 寃곌낵媛믪쓣 �섑뙆�대줈 諛붽씀硫댁꽌 face_descriptors�� �볦븘以���

    return np.array(face_descriptors)

# Compute Saved Face Descriptions - 誘몃━ ���ν븳 �쇨뎬�ㅼ쓽 �몄퐫�⑺븳 �곗씠�� ����

img_paths = Image
#  {
#     'sample1': 'C:/Users/bitcamp/Downloads/Middle_Resolution/19062421/S001/L1/E01/C7.jpg',
#     'sample2':'C:/Users/bitcamp/Downloads/Middle_Resolution/19062431/S001/L1/E01/C7.jpg',
#     'sample3':'C:/Users/bitcamp/Downloads/Middle_Resolution/19062731/S001/L1/E01/C7.jpg'
# }
# �대�吏��� ���μ냼�� ����, �몄퐫�⑸맂 媛믪쓣 ���ν뻽�ㅺ� �명뭼�� �ㅼ뼱�붿쓣 �� �몄퐫�쒕맂 媛믪쓣 爰쇰궡�ㅺ� �멸굅��
descs = { # 怨꾩궛�� 寃곌낵瑜� ���ν븷 蹂���
    'sample1':None
    # 'sample2':None,
    # 'sample3':None
}

for name, img_path in img_paths.items(): # �대�吏� path留뚰겮 猷⑦봽 �뚮㈃�� 
    img_bgr = cv2.imread(img_path) # �대�吏� 濡쒕뱶 
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # bgr�� rgb濡� 諛붽퓞

    _, img_shapes, _ = find_faces(img_rgb) # rgb濡� 諛붽씔 �대�吏��먯꽌 �쇨뎬 李얠븘�� �쒕뱶留덊겕瑜� 諛쏆븘��
    descs[name]=encode_faces(img_rgb, img_shapes)[0] # encode_faces �⑥닔�� �꾩껜 �대�吏��� 媛� �щ엺�� �쒕뱶留덊겕 �ｌ뼱以�/媛� �щ엺�� �대쫫�� 留욊쾶 ���ν빐以�

np.save('teamproject/sample/descs4.npy', descs) # �섑뙆�� �뚯씪濡� ����

print(descs)



# compute input
img_bgr = cv2.imread('C:/Users/bitcamp/Downloads/Middle_Resolution/19062421/S001/L1/E01/C7.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

rects, shapes, _ = find_faces(img_rgb)
descriptors = encode_faces(img_rgb, shapes) # �몄퐫�쒗븳 寃곌낵瑜� descriptors�� 諛쏆븘��

# visualize output / 寃곌낵媛믪쓣 肉뚮젮二쇰뒗 怨쇱젙
fig, ax = plt.subplots(1, figsize=(20,20))
ax.imshow(img_rgb)

for i, desc in enumerate(descriptors): # descriptors留뚰겮 猷⑦봽瑜� �뚭퀬
    found = False
    for name, saved_desc in descs.items(): # �쇨뎬 �몄퐫�쒕맂 媛믪쓣 ���ν븳 descs�� 猷⑦봽 �뚮㈃�� // �꾧� �꾧뎔吏� 鍮꾧탳�섎뒗 遺�遺�
        dist = np.linalg.norm([desc]-saved_desc, axis=1) # distance linear algebra norm  �좏겢由щ뵒�� distance 嫄곕━ 援ы븿/
        
        if dist<0.6: # 0.6 �댄븯�� �� �깅뒫�� �쒖씪 醫뗭븘
            found = True
            # 洹몃━�� 遺�遺�
            text = ax.text(rects[i][0][0], rects[i][0][1],name, # 李얠쑝硫� 洹� �щ엺�� �대쫫�� �대쫫 �⑤씪
                           color='b', fontsize=40, fontweight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=10, foreground='white'),
            path_effects.Normal()])
            
            rect = patches.Rectangle(rects[i][0], # �쇨뎬 遺�遺꾩쓣 �ш컖�뺤쑝濡� 洹몃┝ 
                                     rects[i][1][1] - rects[i][0][1],
                                     rects[i][1][0] - rects[i][0][0],
                                     linewidth=2, edgecolor='w', facecolor='none'
                                     )
            ax.add_patch(rect)

            break
    if not found: # 紐살갼�쇰㈃ unknown�쇰줈 �쒖떆
        ax.text(rects[i][0][0], rects[i][0][1], 'unknown',color='r', fontsize=20, fontweight='bold')
        rect = patches.Rectangle(rects[i][0],
                                 rects[i][1][1] - rects [i][0][1],
                                 rects[i][1][0] - rects [i][0][0],
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

plt.axis('off')
plt.savefig('teamproject/sample/sample5.jpg')
plt.show()
        