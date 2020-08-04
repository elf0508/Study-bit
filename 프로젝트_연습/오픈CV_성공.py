import cv2, dlib, sys 

import numpy as  np


scaler = 0.2

# face_detector 와 landmark predictor 정의

detector = dlib.get_frontal_face_detector()
# dlib에 있는 정면 얼굴 검출기로 입력 사진 img에서 얼굴을 검출하여 faces로 반환

# predictor = dlib.shape_predictor('D:/data/opencv/shape_predictor_68_face_landmarks.dat')


# cap = cv2.VideoCapture('D:\Study-bit\data\opencv/face(1).mp4')
cap = cv2.VideoCapture('D:\Study-bit\data\opencv/face.mp4')
# cv2.VideoCapture() : 비디오 캡쳐 객체를 생성할 수 있다. 
# 안의 숫자는 장치 인덱스(어떤 카메라를 사용할 것인가)이며,
# 1개만 부착되어 있으면 0, 2개 이상이면 첫 웹캠은 0, 두번째 웹캠은 1으로 지정하면된다.

 
while True:

    ret, img = cap.read()   #  이미지를 보여준다.

    if not ret:

        break

 

    img = cv2.resize(img, (int(img.shape[1]*scaler), int(img.shape[0]*scaler)))
    # cv2.resize를 이용해서 이미지의 사이즈 변환

    ori = img.copy()

 

    faces = detector(img)     # left, top, right, bottom 총 4개의 값 반환

    # print(faces)            # rectangles[[(89, 319) (192, 423)]] 

                              # [(face.left(), face.top()) (face.right(), face.bottom())]

    for face in faces:

        img = cv2.rectangle(img, pt1 = (face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255, 255, 0),

                            thickness=2, lineType=cv2.LINE_AA)
        # cv2.rectangle을 이용해서 얼굴이 있는 부분을 박스를 만든다.
    

    cv2.imshow('img', img)  
    # cv2.imshow(tital, image) : title은 윈도우 창의 제목을 의미하며 image는 cv2.imread() 의 return값 

    cv2.waitKey(1)
    # cv2.waitKey()는 키보드 입력을 대기하는 함수로 0이면 key 입력이 있을때까지 무한대기. 