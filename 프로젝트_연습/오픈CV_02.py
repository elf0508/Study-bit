import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

# 이미지 읽고 출력하기

def aidemy_imshow(name, img):
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()

cv2.imshow = aidemy_imshow

img = cv2.imread("./cleansing_data/sample.jpg")

# 이미지 만들어 저장하기

cv2.imshow("sample", img)
img_size = (512, 512)

my_img = np.array([[[0, 0, 255] for _ in range(img_size[1])] for _ in range(img_size[0])], dtype="uint8")

cv2.imshow("sample", my_img)
# cv2.imwrite("my_red_img", my_img)

#########################################

img = cv2.imread("./cleansing_data/sample.jpg")
# size = img.shape

# my_img = img[: size[0] // 2, : size[1] // 3]

my_img = cv2.resize(my_img, (my_img.shape[1] * 2, my_img.shape[0] * 2))
my_img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))

cv2.imshow("sample", my_img)
cv2.imshow("sample", img)

# 회전 및 반전
# cv2.warpAffine : 이미지 회전(아핀 변환)
# cv2.flip(이미지, 대상 축) : 이미지 반전


# cv2.warpAffine() 함수 사용에 필요한 행렬을 만든다.
#                                                    회전의 중심        
mat = cv2.getRotationMatrix2D(tuple(np.array(img.shape[:2]) / 2), 180, 2.0)  # 인수의 배열 : 2배 확대 
#                                                             회전의 각도(180도)

# 회전
my_img = cv2.warpAffine(img, mat, img.shape[:2])
#                       img : 변환 하려는 이미지
#                             mat : 위에서 생성한 행렬
#                                  img.shape[:2] : 위에서 생성한 행렬, 인수의 사이즈


# 반전
# my_img = cv2.flip(img, 0)
my_img = cv2.flip(img, 1)

cv2.imshow("sample", my_img)

# 색조 변환 및 생상 반전

# 색 공간을 변환
# Lab 색 공간으로 변환 : 인간의 시각에 근접하게 설계되어 있다는 장점이 있다.
my_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# COLOR_RGB2GRAY 로 하면 흑백 이미지로 변환된다.
# my_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


cv2.imshow("sample", my_img)

# 이미지의 색상을 반전 시키는 것 : 네거티브 반전
# OpenCV에서의 네거티브 반전은 다음처럼 기술한다.
img = cv2.bitwise_not(img)

# cv2.bitwise()함수는 8bit로 표현된 각 화소의 비트를 조작할 수 있다.
# not 은 각 비트를 반전시킨다.

cv2.imshow("sample", my_img)


# img = cv2.bitwise_not(img) 와 동일한 코드

# for i in range(len(img)):
#     for j in range(len(img[i])):
#         for k in range(len(img[i][j])):
#             img[i][j][k] = 255 - img[i][j][k]

'''

임곗값 처리(이진화)
: 이미지의 용량을 줄이기 위해 일정 이상으로 밝은 것 
혹은 일정 이상으로 어두운 것을 모두 같은 값으로 만드는 것

첫번째 인수 : 처리하는 이미지
두번째 인수 : 임곗값
세번째 인수 : 최댓값(maxvalue)

네번째 인수 

THRESH_BINARY : 픽셀값이 임곗값을 초과하는 경우 해당 픽셀을 maxvalue로 하고, 
                그 외의 경우에는 0(검은색)으로 한다.

THRESH_BINARY_INV : 픽셀값이 임곗값을 초과하는 경우 0으로 설정하고, 
                    그 외의 경우에는 maxvalue로 한다.

THRESH_TRUNC : 픽셀값이 임곗값을 초과하는 경우 임곗값으로 설정하고,
               그 외의 경우에는 변경하지 않는다.

THRESH_TOZERO : 픽셀값이 임곗값을 초과하는 경우 변경하지 않고,
                그 외의 경우에는 0으로 설정한다.

THRESH_TOZERO_INV : 픽셀값이 임곗값을 초과하는 경우 0으로 설정하고, 
                    그 외의 경우에는 변경하지 않는다.

'''

retval, my_img = cv2.threshold(img, 75, 255, cv2.THRESH_TOZERO)
# retval, my_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

cv2.imshow("sample", my_img)


# 마스킹 : 이미지의 일부를 추출하는 것

# 이미지의 흰 부분만 추출

# 두 번째 인수로 0을 지정하면 채널수가 1인 이미지로 변환해서 읽는다.
mask = cv2.imread("./cleansing_data/mask.png", 0)

# 원래 이미지와 같은 크기로 리사이즈한다.
mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

# 세번째 인수로 마스크용 이미지를 선택한다.
my_img = cv2.bitwise_and(img, img, mask = mask)

# 검은 부분만 추출
# retval, my_img = cv2.threshold(mask, 0, 255, cv2.THRESH_TOZERO_INV)
# my_img = cv2.bitwise_and(img, img, mask = mask)

cv2.imshow("sample", my_img)

# 흐림 : 이미지를 흐리게 하려면 픽셀 주위의 n X n개(마스크 크기) 픽셀과의 평균을 취한다.

# 첫번째 인수 : 원본 이미지
# 두번째 인수 : n X n개(마스크 크기)의 n값 (n은 홀수)
# 세번째 인수 : x축 방향의 편차(일반적으로 0으로 지정)

my_img = cv2.GaussianBlur(img, (5, 5), 0)
# my_img = cv2.GaussianBlur(img, (21, 21), 0)

cv2.imshow("sample", my_img)

# 노이즈 제거
my_img = cv2.fastNlMeansDenoisingColored(img)

cv2.imshow("sample", my_img)

# 팽창 및 침식

# 주로 이진 이미지로 처리된다.

# 팽창 : 어떤 한 픽셀을 중심으로 두고, 필터 내의 최댓값을 중심값으로 하는 것 / cv2.dilate()
# 침식 : 어떤 한 픽셀을 중심으로 두고, 최솟값으로 하는 것 / cv2.erode()


# 필터 정의
filt = np.array([[0, 1, 0], 
                 [1, 0, 1],
                 [0, 1, 0]], np.uint8)

# 팽창 처리
my_img = cv2.dilate(img, filt)

cv2.imshow("sample", my_img)

# 침식
cv2.imshow("original", img)
plt.show()
cv2.imshow("sample", my_img)





