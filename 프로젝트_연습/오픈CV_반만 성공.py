import cv2
import numpy as np

# 이미지를 읽고 축소

# wget https://bigsnarf.files.wordpress.com/2017/05/hammer.png    # 흑백
# wget https://i1.wp.com/images.hgmsites.net/hug/2011-volvo-s60_100323431_h.jpg
img = cv2.pyrDown(cv2.imread('D:/Study-bit/img/2011-volvo-s60_100323431_h.jpg', cv2.IMREAD_UNCHANGED))
# img = cv2.pyrDown(cv2.imread('2011-volvo-s60_100323431_h.jpg', cv2.IMREAD_UNCHANGED))


# 임계값 이미지

ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                127, 255, cv2.THRESH_BINARY)

# 등고선을 찾아 외형을 구하다.

contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
#                cv2.CHAIN_APPROX_SIMPLE)

# 각 등고선을 녹색으로
# 빨간색으로 된 minAreaRect
# 파란색의 minEnclosingCircle

for c in contours:    # 윤곽선에서 c의 경우 :
    # 경계 rect를 얻는다
    x, y, w, h = cv2.boundingRect(c)

    # 경계 사각형을 시각화하기 위해 녹색 사각형을 그린다.
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 최소 영역을 수정
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)

    # 모든 좌표 부동 소수점 값을 int로 변환
    box = np.int0(box)

    # 빨간 'nghien'사각형을 그린다.
    cv2.drawContours(img, [box], 0, (0, 0, 255))

    # 마지막으로 최소 둘러싸는 서클을 가져온다.
    (x, y), radius = cv2.minEnclosingCircle(c)

    # 모든 값을 int로 변환
    center = (int(x), int(y))
    radius = int(radius)

    # 파란색으로 원을 그린다
    img = cv2.circle(img, center, radius, (255, 0, 0), 2)

print(len(contours))
cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

cv2.imshow("contours", img)

cv2.imshow("contours", img)

while True:
    key = cv2.waitKey(1)
    if key == 27:   #  ESC key ==> break
        break

cv2.destroyAllWindows()



