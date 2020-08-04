import dlib, cv2, os
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt

# imutils : 이미지 처리 유틸리티
# imutils import face_utils을 사용하면, dlib shape  -->  numpy ndarray로 변환을 안해도된다.

# detector = dlib.get_frontal_face_detector()
# dlib에 있는 정면 얼굴 검출기로 입력 사진 img에서 얼굴을 검출하여 faces로 반환

detector = dlib.cnn_face_detection_model_v1('D:\Study-bit\프로젝트_연습\얼굴인식\mmod_human_face_detector.dat')
# detector = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')
# 얼굴의 위치를 찾는 모델

'''
predictor = dlib.shape_predictor('D:/data/opencv/shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor('D:/Study-bit/프로젝트_연습/얼굴인식/dog_face_detector-master/dog_face_detector-masterlandmarkDetector.dat')
# predictor = dlib.shape_predictor('landmarkDetector.dat')
# 랜드마크를 찾는 모델

 
img_path = 'D:/Study-bit/img_1/Celebrity/아이유'
# img_path = 'img/18.jpg'
filename, ext = os.path.splitext(os.path.basename(img_path))
img = cv2.imread(img_path)   # cv2.imread를 사용해서, 이미지를 불러온다.

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#  cv2.cvtColor는 이미지를 BGR형태로 불러오기 때문에, RGB형태로 바꿔준다.

# img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)

# 첫 이미지를 보여준다.
plt.figure(figsize=(16, 16))
plt.imshow(img)

dets = detector(img, upsample_num_times=1)
# dets 라는 변수에 detector를 사용해서 불러온 이미지의 얼굴의 위치가 들어가게된다.

print(dets)   #  dets 라는 변수는 배열의 형태로 되어있다.

img_result = img.copy()

# for문을 돌면서 이미지에 사각형(d.rect)을 만들어서 보여준다.
for i, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
                                            #  d.confidence 는 몇 %의 확률인지 확인하는 정확도를 나타낸다.
    # 너무 길기때문에 변수를 지정해서 보여준다.
    x1, y1 = d.rect.left(), d.rect.top()
    x2, y2 = d.rect.right(), d.rect.bottom()

    # cv2.rectangle 함수를 사영해서 이미제에 사각형을 그려서 보여준다.
    cv2.rectangle(img_result, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=(255,0,0), lineType=cv2.LINE_AA)
    
plt.figure(figsize=(16, 16))
plt.imshow(img_result)


shapes = []

for i, d in enumerate(dets):
    shape = predictor(img, d.rect)  
    # shape에 6개의 점을 반환해준다 = predictor(이미지와 얼굴의 사각형)을 넣어준다.

    shape = face_utils.shape_to_np(shape)
    # face_utils.shape_to_np 함수로 dlib shape 객체를 numpy array로 변환해준다.
    
    # shape를 나중에도 사용할 예정이므로, [] 형태로 만들어준다.
    for i, p in enumerate(shape):
        shapes.append(shape)

        # 만들어준 shape에 cv2.circle을 이용해서 점을 찍어준다.
        cv2.circle(img_result, center=tuple(p), radius=3, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
        
        # cv2.putText를 이용해서 이미지에 글씨를 그려준다. (숫자를 나타냈다.)
        cv2.putText(img_result, str(i), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

img_out = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)
cv2.imwrite('img/%s_out%s' % (filename, ext), img_out)
plt.figure(figsize=(16, 16))
plt.imshow(img_result)

from math import atan2, degrees

# overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    img_to_overlay_t = cv2.cvtColor(img_to_overlay_t, cv2.COLOR_BGRA2RGBA)
    bg_img = background_img.copy()
    # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGB2RGBA)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGBA2RGB)

    return bg_img

def angle_between(p1, p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    return degrees(atan2(yDiff, xDiff))


img_result2 = img.copy()

horns = cv2.imread('img/horns2.png',  cv2.IMREAD_UNCHANGED)
horns_h, horns_w = horns.shape[:2]

nose = cv2.imread('img/nose.png',  cv2.IMREAD_UNCHANGED)

for shape in shapes:
    horns_center = np.mean([shape[4], shape[1]], axis=0) // [1, 1.3]
    horns_size = np.linalg.norm(shape[4] - shape[1]) * 3
    
    nose_center = shape[3]
    nose_size = horns_size // 4

    angle = -angle_between(shape[4], shape[1])
    M = cv2.getRotationMatrix2D((horns_w, horns_h), angle, 1)
    rotated_horns = cv2.warpAffine(horns, M, (horns_w, horns_h))

    img_result2 = overlay_transparent(img_result2, nose, nose_center[0], nose_center[1], overlay_size=(int(nose_size), int(nose_size)))
    try:
        img_result2 = overlay_transparent(img_result2, rotated_horns, horns_center[0], horns_center[1], overlay_size=(int(horns_size), int(horns_h * horns_size / horns_w)))
    except:
        print('failed overlay image')

img_out2 = cv2.cvtColor(img_result2, cv2.COLOR_RGB2BGR)
cv2.imwrite('img/%s_out2%s' % (filename, ext), img_out2)
plt.figure(figsize=(16, 16))
plt.imshow(img_result2)
'''
