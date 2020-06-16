import cv2
import os
'''
scale factor는 1에 가까울수록 인식율이 좋지만 
그만큼 느려짐(그만큼 많은  
'''
SF  = 1.02
'''
내부 알고리즘에서 최소한 검출된 횟수이상 되야 인식
0이면 무수한 오 검출이 되고
1이면 한번 이상 검출된 곳만 인식된다.
값이 높아질수록 오인식율은 줄지만 그만큼 
인식율이 떨어진다.
'''
N = 2
'''
검출하려는 이미지의 최소 사이즈
이 값보다 작은 이미지는 무시 
'''
MS=(100,100)

# 고양이 얼굴 인식용 haarcascade 파일 위치 
cascade = '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalcatface.xml'

# 고양이 얼굴 인식 cascade 생성 
face_cascade = cv2.CascadeClassifier(cascade)

# 얼굴 검출용 이미지 
catPic = './cat1/6.jpg'

# 얼굴 검출 함수 
def detectCatFace(imgPath):
    # 이미지 불러오기 
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR);
    # 회색으로 변경 
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 얼굴 검출 
    faces = face_cascade.detectMultiScale(grayImg, scaleFactor=SF, minNeighbors=N, minSize=MS)
    # 검출된 얼굴 개수 출력 
    print("The number of images found is : " + str(len(faces)))    
    # 검출된 얼굴 위치에 녹색 상자그리기 
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    # 검출된 얼굴(녹색 상자가 그려진) 이미지 데이터를 리턴
    return img    

# 얼굴 검출 함수 호출 
img = detectCatFace(catPic)
# 검출된 이미지가 있다면 화면에 표시 
if len(img) != 0:
    cv2.imshow('Face',img)

# 아무키나 눌릴때까지 대기 
cv2.waitKey(0)

cv2.destroyWindow('Face')