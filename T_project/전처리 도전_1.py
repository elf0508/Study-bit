import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.decomposition import PCA


img_C7 = cv2.imread('T_project/file/L4.jpg/C7.jpg', cv2.IMREAD_GRAYSCALE)  # 흑백 이미지 로드

plt.imshow(img_C7, cmap = 'gray')   # 이미지 출력
plt.axis('off')
plt.show()

type(img_C7)  # 데이터 타입 확인
img_C7        # 이미지 데이터 확인
img_C7.shape    # 차원 확인(해상도) / (230, 346)



# 컬러로 이미지 로드
img_C7_bgr = cv2.imread('T_project/file/L4.jpg/C7.jpg', cv2.IMREAD_COLOR)

img_C7_bgr = cv2.cvtColor(img_C7_bgr, cv2.COLOR_BGR2RGB)   # RGB로 변환
plt.imshow(img_C7_bgr)   # 이미지 출력
plt.axis('off')
plt.show()

# 이미지 자르기
# 이미지 주변을 제거해서 차원을 줄여준다.
# 2차원 넘파이로 저장된다.

img_C7 = cv2.imread('T_project/file/L4.jpg/C7.jpg',  cv2.IMREAD_COLOR)
# img_C7 = cv2.imread('T_project/file/L4.jpg/C7.jpg', cv2.IMREAD_GRAYSCALE)
img_C7_croppen = img_C7[:, -225:]
# img_C7_croppen = img_C7[:, :125]

plt.imshow(img_C7_croppen, cmap='gray')
plt.axis('off')
plt.show()

# 배경 제거
# 이미지의 전경만 분리해내려면 원하는 전경 주위에 사각형 박스를 그리고 그랩컷 알고리즘을 실행한다.
# 그랩컷은 사각형 밖에 있는 모든 것이 배경이라고 가정하고, 이 정보를 사용하여 사각형 안에 있는 배경을 찾는다.
# 검은색 영역은 배경이라고 확실하게 가정한 사각형의 바깥쪽 영역이며, 
# 회색 영역은 그랩컷이 배경이라고 생각하는 영역이고 흰색 영역은 전경이다.

img_C7_bgr = cv2.imread('T_project/file/L4.jpg/C7.jpg') # 이미지 로드
img_C7_rgb = cv2.cvtColor(img_C7_bgr, cv2.COLOR_BGR2RGB) # RGB로 변환

rectangle = (0, 56, 256, 150) # 사각형 좌표: 시작점의 x, 시작점의 y, 너비, 높이

mask = np.zeros(img_C7_rgb.shape[:2], np.uint8) # 초기 마스크를 만듭니다.

bgdModel = np.zeros((1, 65), np.float64) # grabCut에 사용할 임시 배열을 만든다.
fgdModel = np.zeros((1, 65), np.float64)

# grabCut 실행
cv2.grabCut(img_C7_rgb, # 원본 이미지
            mask, # 마스크
            rectangle, # 사각형
            bgdModel, # 배경을 위한 임시 배열
            fgdModel, # 전경을 위한 임시 배열
            5, # 반복 횟수
            cv2.GC_INIT_WITH_RECT) # 사각형을 사용한 초기화
            
# 배경인 곳은 0, 그외에는 1로 설정한 마스크를 만듭니다.
mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

# 이미지에 새로운 마스크를 곱해 배경을 제외합니다.
img_C7_rgb_nobg = img_C7_rgb * mask_2[:, :, np.newaxis]
plt.imshow(img_C7_rgb_nobg)
plt.axis("off") # 이미지 출력
plt.show()

plt.imshow(mask, cmap='gray')
plt.axis("off") # 마스크 출력
plt.show()

plt.imshow(mask_2, cmap='gray')
plt.axis("off") # 마스크 출력
plt.show()

# 캐니(Canny) 경계선 감지

# 경계선 감지는 컴퓨터 비전의 주요 관심 대상이며, 경계선은 많은 정보가 담긴 영역이다.
# 경계선 감지를 사용하여 정보가 적은 영역을 제거하고, 대부분의 정보가 담긴 이미지 영역을 구분할 수 있다.

# https://carstart.tistory.com/187?category=209768

# 컴퓨터 비전의 목적 : 이미지가 나타내는 이야기를 해석하는 컴퓨터 프로그램으로 만드는 것
# 한마디로 우리가 아닌 컴퓨터가 이미지가 나타내는 이야기를 해석하는 것

# 어떤 분야에서 사용 되는지??
# 1. 구글 어스의 3D 지도 모델링
# 2. 여러 장의 사진을 겹쳐서 하나의 파노라마로 만들어주는 것
# 3. 보안을 위한 얼굴, 홍재, 지문 인식 등
# 4. 자동차의 다양한 기술들 - 차선검출, 졸음 방지 등 다양한 곳에서 사용되고 있다.


# 예제 실습은 낮은 임곗값과 높은 임곗값을 이미지 중간 픽셀 강도의 1표준편차 아래 값과 위 값으로 설정하였다.

img_C7_gray = cv2.imread("T_project/file/L4.jpg/C7.jpg", cv2.IMREAD_GRAYSCALE) 
median_intensity = np.median(img_C7_gray) # 픽셀 강도의 중간값을 계산

# 중간 픽셀 강도에서 위아래 1 표준 편차 떨어진 값을 임계값으로 지정한다.
lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))

# 캐니 경계선 감지기를 적용한다.
img_C7_canny = cv2.Canny(img_C7_gray, lower_threshold, upper_threshold)

plt.imshow(img_C7_canny, cmap="gray")
plt.axis("off") # 이미지 출력
plt.show()
