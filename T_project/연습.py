import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.decomposition import PCA

def aidemy_imshow(name, img):

    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])

    plt.imshow(img)
    plt.show()

cv2.imshow = aidemy_imshow


def aidemy_imshow(name, img):
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    plt.imshow(img)
    plt.show()

# cv2.imshow = aidemy_imshow

# OpenCV의 imread를 사용하여 전처리를 위한 이미지를 로드할 수 있다.
# imread() : 이미지를 넘파이 배열(행렬)로 변환한다. 행렬의 각 원소는 개별 픽셀에 해당한다.

img_C7 = cv2.imread('T_project/file/L4.jpg/C7.jpg', cv2.IMREAD_GRAYSCALE)  # 흑백 이미지 로드

plt.imshow(img_C7, cmap = 'gray')   # 이미지 출력
plt.axis('off')
plt.show()

type(img_C7)  # 데이터 타입 확인
img_C7        # 이미지 데이터 확인
img_C7.shape    # 차원 확인(해상도) / (230, 346)

# 컬러로 이미지 로드
img_C7_bgr = cv2.imread('T_project/file/L4.jpg/C7.jpg', cv2.IMREAD_COLOR)
px = img_C7_bgr[100, 100]
print(px)       # [6 4 3]

img_C7_bgr[0, 0]   # 픽셀 확인
img_C7_bgr = cv2.cvtColor(img_C7_bgr, cv2.COLOR_BGR2RGB)   # RGB로 변환
plt.imshow(img_C7_bgr)   # 이미지 출력
plt.axis('off')
plt.show()

# 이미지 크기 변경
# resize() : 이미지 크기를 변경
# 전처리로서 이미지 크기를 변경이 필요한 이유는 이미지들은 제각기 다양한 크기를 가지며, 
# 특성으로 사용하려면 동일한 차원으로 만들어줘야 하기 때문이다.
# 이미지 행렬에 정보를 담고 있기 때문에 이미지 크기를 표준화하게 되면,
# 이미지의 행렬 크기와 거기에 담긴 정보도 줄어든다.  --> 메모리의 사용량을 줄일수있다.
# 가장 많이 사용하는 이미지 크기 : 32x32, 64x64, 96x96, 245x245


img_C7 = cv2.imread('T_project/file/L4.jpg/C7.jpg', cv2.IMREAD_GRAYSCALE)
# img_C1_50x50 = cv2.resize(img_C1, (50, 50))
# plt.imshow(img_C1_50x50, cmap='gray')

img_C7_10x10 = cv2.resize(img_C7, (10, 10))
plt.imshow(img_C7_10x10, cmap='gray')
plt.axis('off')
plt.show

# 이미지 자르기
# 이미지 주변을 제거해서 차원을 줄여준다.
# 2차원 넘파이로 저장된다.

img_C7 = cv2.imread('T_project/file/L4.jpg/C7.jpg', cv2.IMREAD_GRAYSCALE)
img_C7_croppen = img_C7[:, -225:]
# img_C7_croppen = img_C7[:, :125]

plt.imshow(img_C7_croppen, cmap='gray')
plt.axis('off')
plt.show()

# 이미지 투명도 처리
# 이미지를 흐리게 하려면, 각 픽셀을 주변 픽셀의 평균값으로 변환한다.
# 커널 : (1) 주변 픽셀에 수행되는 연산을 수학적으로 표현한 것
#        (2) 커널CPA와 벡터 머신이 사용하는 비선형 함수
#        (3) 신경망의 가중치
# Meanshift 알고리즘에서는 샘플의 영향 범위를 커널이라 부른다.
# 커널의 크기는 흐림의 정도를 결정한다.
# 커널이 클수록 이미지가 부드러워진다.
# 커널 크기 : (너비, 높이)로 지정한다.
# blur 함수는 각 픽셀에 커널 개수의 역수를 곱하여, 모두 더한다.(이 값이 중앙 픽셀의 값이 된다.)

img_C7 = cv2.imread('T_project/file/L4.jpg/C7.jpg', cv2.IMREAD_GRAYSCALE)

# 각 픽셀 주변의 5x5커널 평균값으로 이미지를 흐리게 한다.
img_C7_blurry = cv2.blur(img_C7, (5, 5))

plt.imshow(img_C7_blurry, cmap='gray')
plt.axis('off')
plt.show()

# 커널 크기의 영향을 강조하기 위헤 100x100 커널로 같은 이미지를 흐리게 만든다.
img_C7_very_blurry = cv2.blur(img_C7, (100, 100))
plt.imshow(img_C7_very_blurry, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

kernel = np.ones((5, 5)) / 25.0  # 커널을 만든다.
kernel   # 커널 확인
img_C7_kernel = cv2.filter2D(img_C7, -1, kernel)   # 커널을 적용
plt.imshow(img_C7_kernel, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()


img_C7_very_blurry = cv2.GaussianBlur(img_C7, (5, 5), 0)   # 가우시안 블러 적용
plt.imshow(img_C7_very_blurry, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()


# GaussianBlur()의 세번째 매개변수 : X축(너비) 방향의 표준편차
# 0으로 지정하면 ((너비-1)*0.5-1)*0.3+0.8와 같이 계산한다.
# Y축 방향의 표준편차는 기본값이 0이다.
# 가우시안 블러에 사용한 커널은 각 축 방향으로 가우시안 분포를 따르는 1차원 배열을 만든 다음 외적하여 생성한다.
# getGaussianKernel()를 사용하여 1차원 배열을 만들고, 넘파이 outer 함수로 외적을 계산할 수 있다.
# filter2D()의 두번째 매개변수는 픽셀값의 범위를 지정하는 것으로 -1이면 입력과 동일한 범위를 유지한다.

gaus_vector = cv2.getGaussianKernel(5, 0)
gaus_vector
gaus_kernel = np.outer(gaus_vector, gaus_vector)  # 벡터를 외적하여 커널을 만든다.
gaus_kernel

# filter2D()로 커널을 이미지에 직접 적용하여 비슷한 흐림 효과를 만들 수 있다.
img_C7_kernel = cv2.filter2D(img_C7, -1, gaus_kernel) # 커널을 적용
plt.imshow(img_C7_kernel, cmap="gray")
plt.xticks([])
plt.yticks([])  
plt.show()

# 이미지 선명하게 
# 대상 픽셀을 강조하는 커널을 만들고, filter2D를 사용하여 이미지에 커널을 적용한다.
# 중앙 픽셀을 부각하는 커널을 만들면 이미지의 경계선에서 대비가 더욱 두드러지는 효과가 생긴다.

img_C7 = img_C7 = cv2.imread('T_project/file/L4.jpg/C7.jpg', cv2.IMREAD_GRAYSCALE) 

kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]]) # 커널을 만듭니다.

# 이미지를 선명하게 만듭니다.
img_C7_sharp = cv2.filter2D(img_C7, -1, kernel)

plt.imshow(img_C7_sharp, cmap="gray")
plt.axis("off") 
plt.show()

# 이미지 대비 높이기
# 히스토그램 평활화 : 객체의 형태가 두드러지도록 만들어주는 이미지 처리 도구
# Y는 루마(luma) 또는 밝기이고, U와 V는 컬러를 나타낸다.
# 흑백 이미지에는 OpenCV의 equalizeHist()를 바로 적용할 수 있다.
# 히스토그램 평활화는 픽셀값의 범위가 커지도록 이미지를 변환한다.
# 히스토그램 평활화는 관심 대상을 다른 객체나 배경과 잘 구분되도록 만들어준다.

img_C7 = cv2.imread('T_project/file/L4.jpg/C7.jpg', cv2.IMREAD_GRAYSCALE) 
img_C7_enhanced = cv2.equalizeHist(img_C7) # 이미지 대비를 향상시킵니다.
plt.imshow(img_C7_enhanced, cmap="gray")
plt.axis("off") # 이미지 출력
plt.show()

img_C7_bgr = cv2.imread('T_project/file/L4.jpg/C7.jpg') # 이미지 로드
img_C7_yuv = cv2.cvtColor(img_C7_bgr, cv2.COLOR_BGR2YUV) # YUV로 변경합니다.
img_C7_yuv[:, :, 0] = cv2.equalizeHist(img_C7_yuv[:, :, 0]) # 히스토그램 평활화를 적용
img_C7_rgb = cv2.cvtColor(img_C7_yuv, cv2.COLOR_YUV2RGB) # RGB로 바꿉니다.
plt.imshow(img_C7_rgb)
plt.axis("off") # 이미지 출력
plt.show()


# 색상 구분
# 이미지에서 한 색상을 구분하려면, 색 범위를 정의하고 이미지에 마스크를 적용한다.
# 이미지를 HSV(색상, 채도, 명도)로 변환 -> 격리시킬 값의 범위를 정의 -> 이미지에 적용할 마스크를 만든다.
# (마스크의 흰색 영역만 유지)
#  bitwise_and()는 마스크를 적용하고 원하는 포맷으로 변환

img_C7_bgr = cv2.imread('T_project/file/L4.jpg/C7.jpg') # 이미지 로드
img_C7_hsv = cv2.cvtColor(img_C7_bgr, cv2.COLOR_BGR2HSV) # BGR에서 HSV로 변환
lower_blue = np.array([50,100,50]) # HSV에서 파랑 값의 범위를 정의
upper_blue = np.array([130,255,255])
mask = cv2.inRange(img_C7_hsv, lower_blue, upper_blue) # 마스크를 만듭니다.
img_C7_bgr_masked = cv2.bitwise_and(img_C7_bgr, img_C7_bgr, mask=mask) # 이미지에 마스크를 적용
img_C7_rgb = cv2.cvtColor(img_C7_bgr_masked, cv2.COLOR_BGR2RGB) # BGR에서 RGB로 변환

plt.imshow(img_C7_rgb)
plt.axis("off") # 이미지를 출력
plt.show()

plt.imshow(mask, cmap='gray')
plt.axis("off") # 마스크 출력
plt.show()

# 이미지 이진화
# 이미지 이진화(임계처리)thresholding은 어떤 값보다 큰 값을 가진 픽셀을 흰색으로 만들고,
# 작은 값을 가진 픽셀은 검은색으로 만드는 과정이다.

# 적응적 이진화(임계처리)adaptive thresholding은 픽셀의 임계값이 주변 픽셀의 강도에 의해 결정된다.
# 이진화는 이미지 안의 영역 마다 빛 조건이 달라질 때 도움이 된다.
# adaptiveThreshold()의 max_output_value매개변수는 출력 픽셀 강도의 최대값을 결정
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C는 픽셀의 임계값을 주변 픽셀 강도의 가중치 합으로 설정
# cv2.ADAPTIVE_THRESH_MEAN_C는 픽셀의 임계값을 주변 픽셀의 평균으로 설정

img_C7_grey = cv2.imread("T_project/file/L4.jpg/C7.jpg", cv2.IMREAD_GRAYSCALE) 
max_output_value = 255
neighborhood_size = 99
subtract_from_mean = 10
img_C7_binarized = cv2.adaptiveThreshold(img_C7_grey, max_output_value,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                         neighborhood_size, subtract_from_mean) # 적응적 임계처리를 적용
plt.imshow(img_C7_binarized, cmap="gray")
plt.axis("off") # 이미지 출력
plt.show()


# cv2.ADAPTIVE_THRESH_MEAN_C를 적용.
img_C7_mean_threshold = cv2.adaptiveThreshold(img_C7_grey,
                                             max_output_value,
                                             cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY,
                                             neighborhood_size,
                                             subtract_from_mean)
plt.imshow(img_C7_mean_threshold, cmap="gray")
plt.axis("off") # 이미지를 출력
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

# 경계선 감지
# 캐니(Canny) 경계선 감지기와 같은 경계선 감지 기술 사용
# 경계선 감지는 컴퓨터 비전의 주요 관심 대상이며, 경계선은 많은 정보가 담긴 영역이다.
# 경계선 감지를 사용하여 정보가 적은 영역을 제거하고, 대부분의 정보가 담긴 이미지 영역을 구분할 수 있다.
# 캐니 감지기는 그레이디언트 임계값의 저점과 고점을 나타내는 두 매개변수가 필요하다.
# 낮은 임계값과 높은 임계값 사이의 가능성 있는 경계선 픽셀은 약한 경계선 픽셀로 간주된다
# OpenCV의 Canny 함수는 낮은 임곗값과 높은 임곗값이 필수 매개변수이다.
# Canny를 전체 이미지 모음에 적용하기 전에 몇 개의 이미지를 테스트하여 
# 낮은 임계값과 높은 임곗값의 적절한 쌍을 찾는 것이 좋은 결과를 만든다.
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

 

# 모서리 감지
# cornerHarris - 해리스 모서리 감지의 OpenCV 구현
# 해리스 모서리 감지기는 두 개의 경계선이 교차하는 지점을 감지하는 방법으로 사용된다.
# 모서리는 정보가 많은 포인트이다.
# 해리스 모서리 감지기는 윈도(이웃, 패치)안의 픽셀이 작은 움직임에도 크게 변하는 윈도를 찾는다.
# cornerHarris 매개변수 block_size : 각 픽셀에서 모서리 감지에 사용되는 이웃 픽셀 크기
# cornerHarris 매개변수 aperture : 사용하는 소벨 커널 크기

img_C7_bgr = cv2.imread("T_project/file/L4.jpg/C7.jpg") # 흑백 이미지 로드
img_C7_gray = cv2.cvtColor(img_C7_bgr, cv2.COLOR_BGR2GRAY)
img_C7_gray = np.float32(img_C7_gray)

block_size = 2 # 모서리 감지 매개변수를 설정
aperture = 29
free_parameter = 0.04

detector_responses = cv2.cornerHarris(img_C7_gray,
                                      block_size,
                                      aperture,
                                      free_parameter) # 모서리를 감지
detector_responses = cv2.dilate(detector_responses, None) # 모서리 표시를 부각

# 임계값보다 큰 감지 결과만 남기고 흰색으로 표시.
threshold = 0.02
img_C7_bgr[detector_responses >
          threshold *
          detector_responses.max()] = [255,255,255]

img_C7_gray = cv2.cvtColor(img_C7_bgr, cv2.COLOR_BGR2GRAY) # 흑백으로 변환

plt.imshow(img_C7_gray, cmap="gray")
plt.axis("off") # 이미지 출력
plt.show()

# 가능성이 높은 모서리를 출력.
plt.imshow(detector_responses, cmap='gray')
plt.axis("off")
plt.show()

img_C7_bgr = cv2.imread('T_project/file/L4.jpg/C7.jpg')
img_C7_gray = cv2.cvtColor(img_C7_bgr, cv2.COLOR_BGR2GRAY)

# 감지할 모서리 개수
corners_to_detect = 10
minimum_quality_score = 0.05
minimum_distance = 25

corners = cv2.goodFeaturesToTrack(img_C7_gray,
                                  corners_to_detect,
                                  minimum_quality_score,
                                  minimum_distance) # 모서리를 감지
corners = np.float32(corners)

for corner in corners:
    x, y = corner[0]
    cv2.circle(img_C7_bgr, (x,y), 10, (255,255,255), -1) # 모서리마다 흰 원을 그립니다.
    
img_C7_rgb = cv2.cvtColor(img_C7_bgr, cv2.COLOR_BGR2GRAY) # 흑백 이미지로 변환
plt.imshow(img_C7_rgb, cmap='gray'), plt.axis("off") # 이미지를 출력
plt.show()

# 머신러닝 특성 만들기
# 이미지를 머신러닝에 필요한 샘플로 변환하려면 넘파이의 flatten()을 사용한다.
# Flatten()은 이미지 데이터가 담긴 다차원 배열을 샘플값이 담긴 벡터로 변환
# 이미지가 흑백일 때 각 픽셀은 하나의 값으로 표현된다.
# 컬럼 이미지라면 각 픽셀이 하나의 값이 아니라 여러 개의 값으로 표현된다.
# 이미지의 모든 픽셀이 특성이 되기 때문에 이미지가 커질수록 특성의 개수도 크게 늘어난다.

img_C7 = cv2.imread("T_project/file/L4.jpg/C7.jpg", cv2.IMREAD_GRAYSCALE)
img_C7_10x10 = cv2.resize(img_C7, (10, 10)) # 이미지를 10x10 픽셀 크기로 변환
img_C7_10x10.flatten() # 이미지 데이터를 1차원 벡터로 변환

plt.imshow(img_C7_10x10, cmap="gray")
plt.axis("off")
plt.show()

img_C7_10x10.shape
img_C7_10x10.flatten().shape

img_C7_color = cv2.imread("T_project/file/L4.jpg/C7.jpg", cv2.IMREAD_COLOR) # 컬러 이미지로 로드
img_C7_color_10x10 = cv2.resize(img_C7_color, (10, 10)) # 이미지를 10 × 10 픽셀 크기로 변환
img_C7_color_10x10.flatten().shape # 이미지 데이터를 1차원 벡터로 변환하고 차원을 출력

img_C7_256x256_gray = cv2.imread("T_project/file/L4.jpg/C7.jpg", cv2.IMREAD_GRAYSCALE) # 흑백 이미지로 로드
img_C7_256x256_gray.flatten().shape # 이미지 데이터를 1차원 벡터로 변환하고 차원을 출력

img_C7_256x256_color = cv2.imread("T_project/file/L4.jpg/C7.jpg", cv2.IMREAD_COLOR) # 컬러 이미지로 로드
img_C7_256x256_color.flatten().shape # 이미지 데이터를 1차원 벡터로 변환하고 차원을 출력

# 평균 색을 특성으로 인코딩
# 이미지의 각 픽셀은 여러 컬러 채널(빨간, 초록, 파랑)의 조합으로 표현되며, 채널의 평균값을 계산하여 이미지의
# 평균 컬러를 나타내는 세 개의 컬럼 특성을 만든다.

# BGR 이미지로 로드
img_C7_bgr = cv2.imread("T_project/file/L4.jpg/C7.jpg", cv2.IMREAD_COLOR)
channels = cv2.mean(img_C7_bgr) # 각 채널의 평균을 계산

# 파랑과 빨강을 바꿉니다(BGR에서 RGB로 만듭니다)
observation = np.array([(channels[2], channels[1], channels[0])])
observation # 채널 평균 값을 확인
plt.imshow(observation)
plt.axis("off") # 이미지를 출력
plt.show()


# 컬러 히스토그램을 특성으로 인코딩
# 이미지의 각 픽셀은 여러 컬러 채널(빨간, 초록, 파랑)의 조합으로 표현되며, 채널의 평균값을 계산하여 이미지의
# 평균 컬러를 나타내는 세 개의 컬럼 특성을 만든다.

img_C7_bgr = cv2.imread("T_project/file/L4.jpg/C7.jpg", cv2.IMREAD_COLOR)
img_C7_rgb = cv2.cvtColor(img_C7_bgr, cv2.COLOR_BGR2RGB)# RGB로 변환
features = [] # 특성 값을 담을 리스트
colors = ("r","g","b") # 각 컬러 채널에 대해 히스토그램을 계산

# 각 채널을 반복하면서 히스토그램을 계산하고 리스트에 추가
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([img_C7_rgb], # 이미지
                             [i], # 채널 인덱스
                             None, # 마스크 없음
                             [256], # 히스토그램 크기
                             [0,256]) # 범위
    features.extend(histogram)
    
observation = np.array(features).flatten() # 샘플의 특성 값으로 벡터를 만듭니다.
observation[0:5]

img_C7_rgb[0,0] # RGB 채널 값을 확인

import pandas as pd

data = pd.Series([1, 1, 2, 2, 3, 3, 3, 4, 5]) # 예시 데이터
data.hist(grid=False) # 히스토그램을 출력
plt.show()

colors = ("r","g","b") # 각 컬러 채널에 대한 히스토그램을 계산
# 컬러 채널을 반복하면서 히스토그램을 계산하고 그래프를 그립니다.
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([img_C7_rgb], # 이미지
                             [i], # 채널 인덱스
                             None, # 마스크 없음
                             [256], # 히스토그램 크기
                             [0,256]) # 범위
    plt.plot(histogram, color = channel)
    plt.xlim([0,256])
    
plt.show() # 그래프를 출력