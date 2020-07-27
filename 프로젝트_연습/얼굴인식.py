import dlib
import cv2


# dlib에 있는 정면 얼굴 검출기로 입력 사진 img에서 얼굴을 검출하여 faces로 반환
face_detector = dlib.get_frontal_face_detector()
img = cv2.imread("harry.jpg")
faces = face_detector(img)

# 결과 출력 & 이미지에 해당 검출 부분들을 빨간색(bgr순) 박스로 표시
print("{} faces are detecred.".format(len(faces)))

for f in faces:

    print("left, top, right, bottom : ", f.left(), f.top(), f.right(), f.bottom())
cv2.rectangle(img, (f.left(), f.top(), f.right(), f.bottom()), (0,0,255), 2)

win = dlib.image_window()
win.set_image(img)
win.add_overlay(faces)
cv2.imwrite("output.jpg", img)   # 결과를 output.jpg로 저장