import os
import time
import warnings ; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'D:/Study-bit/dacon/comp4/'
# path = 'C:/Users/bitcamp/Desktop/dacon/'
os.chdir(path)  
# 지정된 경로에 현재 작업 디렉토리를 변경하는데 사용한다.
# 경로 - 새 경로로 전환한다.

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor


### 데이터 전처리
## Data Cleansing & Pre-Processing
## 데이터 정제 & Pre-Processing

def grap_year(data):        # 함수 정의 그룹_년(data)
    data = str(data)        # data = 문자열로 가져온다
    return int(data[:4])    # 정수형(data[처음부터 4번째까지]) --> 년도 : 2019 - 2020

def grap_month(data):       # 함수 정의 그룹_월(data)
    data = str(data)        # data = 문자열로 가져온다
    return int(data[4:])    # 정수형(data[4번째부터 끝까지]) --> 월 : 01 - 03

## 날짜 처리

data = pd.read_csv(path + '201901-202003.csv')  # 날짜 csv를 판다스로 읽어서, data에 대입한다.
data = data.fillna('')

# lambda식을 이용해서 '년도'와 '월'을 새롭게 정의한다.

data['year'] = data['REG_YYMM'].apply(lambda x: grap_year(x))  
data['month'] = data['REG_YYMM'].apply(lambda x: grap_month(x))
data = data.drop(['REG_YYMM'], axis = 1)

## 데이터 정제

df = data.copy()   # 원본 데이터는 그대로 두고, 값을 복사하는 형태로 가져온다.
df = df.drop(['CARD_CCG_NM', 'HOM_CCG_NM'], axis = 1) # 불필요한 부분 삭제

# 새로운 칼럼을 만든다.

columns = ['CARD_SIDO_NM', 'STD_CLSS_NM', 'HOM_SIDO_NM', 'AGE', 'SEX_CTGO_CD', 'FLC', 'year', 'month']

# 새로 만든 컬럼을 그룹으로 묶어서, 같은 값들은 합쳐준다.

df = df.groupby(columns).sum().reset_index(drop = False)

# 예) 여자 7   여자 40   여자 15  여자 100  / 남자 8  남자 10  남자 80 일때
# 하나로 묶어서 값을 합쳐준다(정리). --> 여자 162 / 남자 98

## 인코딩 : 글자를 숫자로 바꾸는 것 / 유니코드를 바이트 열로 변환하는 것

dtypes = df.dtypes
encoders = {}
for column in df.columns:  # 새로 만든 컴럼들을 인코딩해서 컬럼에 대입한다.
    if str(dtypes[column]) == 'object':  # 객체가 존재한다면, 문자열로 바꾼다.
        encoder = LabelEncoder()         # LabelEncoder로 인코딩한다. 
        encoder.fit(df[column])          # 컬럼을 인코딩
        encoders[column] = encoder       # 문자열로 인코딩 한 컬럼을 [컬럼]에 대입한다.

# LabelEncoder : 문자를 0부터 시작하는 정수형 숫자로 바꿔준다. 

df_num = df.copy()     # 원본 데이터는 그대로 두고, 값을 복사하는 형태로 가져온다.
for column in encoders.keys():    # 문자열로 바꾼 컬럼들의 key값을 가져온다.
    encoder = encoders[column]
    df_num[column] = encoder.transform(df[column])  

    # 문자열로 바꾼 값들을 다시 '숫자'로 바꿔준다.(encoder.transform)
    # 바꾼 값들을 df_num[column]에 대입한다.

### 변수 선택 및 모델 구축
## feature, target 설정

train_num = df_num.sample(frac = 1, random_state = 0) 

# dataframe에 랜덤하게 몇몇 데이터만 뽑고 싶을 때가 있다.
# 이 때 sample function을 사용하면 된다.
# sample(frac = 숫자)을 입력하면 전체 row에서 몇%의 데이터를 return할 것인지 정할 수 있다.
# sample(frac = 1) --> 모든 데이터를 반환

train_features = train_num.drop(['CSTMR_CNT', 'AMT', 'CNT'], axis = 1)
train_target = np.log1p(train_num['AMT'])

# np.log1p : (요소 단위로) 입력 어레이에 대해 자연로그 log(1 + x) 값을 반환
# np.log1p == StandardScaler 와 같은 역할을 한다.
# StandardScaler : 평균과 표준편차 사용 / 평균을 제거하고, 데이터를 단위 분산으로 조정한다. 
# 데이터 값을 0에 맞춰준다.

### 모델 학습 및 검증
## Model Tuning & Evaluation
# 훈련

model = XGBRegressor(n_estimators = 150,    # 클수록 좋다 / 단점 : 메모리 많이 차지, 기본값 = 100 / 나무 100개
                     learning_rate = 0.1,   # 학습률 디폴트 값/ 상당히 중요하다
                     max_depth = 20,
                     objective = 'reg:linear',
                     colsample_bytree = 0.7,
                     colsample_bylevel = 0.7,
                     importance_type = 'gain',
                     random_state = 18,
                     n_jobs = -1)     # 병렬처리

model.fit(train_features, train_target)

### 결과 및 결언
## Conclusion & Discussion
# 예측 템플릿 만들기

CARD_SIDO_NMs = df_num['CARD_SIDO_NM'].unique() # unique(): 배열 내 중복된 원소 제거 후 유일한 원소를 정렬하여 반환
STD_CLSS_NMs = df_num['STD_CLSS_NM'].unique()   # 예) x = [1,1,2,3,4,4,5] 
HOM_SIDO_NMs = df_num['HOM_SIDO_NM'].unique()   # x.unique --> x = [1,2,3,4,5]
AGEs = df_num['AGE'].unique()
SEX_CTGO_CDs = df_num['SEX_CTGO_CD'].unique()
FLCs = df_num['FLC'].unique()
years = [2020]
months = [4, 7]

tmp = []
for CARD_SIDO_NM in CARD_SIDO_NMs:
    for STD_CLSS_NM in STD_CLSS_NMs:
        for HOM_SIDO_NM in HOM_SIDO_NMs:
            for AGE in AGEs:
                for SEX_CTGO_CD in SEX_CTGO_CDs:
                    for FLC in FLCs:
                        for year in years:
                            for month in months:
                                tmp.append([CARD_SIDO_NM, STD_CLSS_NM, HOM_SIDO_NM, AGE, SEX_CTGO_CD, FLC, year, month])
tmp = np.array(tmp)
tmp = pd.DataFrame(data = tmp,
                   columns = train_features.columns)

# 예측
pred = model.predict(tmp)  # 새로 만든 tmp 데이터로 pred에 대입한다.
pred = np.expm1(pred)    # np.log1p로 0에 맞춰 조정 한 값들을 np.expm1를 사용해서 다시 원래의 데이값으로 만든다.
tmp['AMT'] = np.round(pred, 0)  #  np.round(x, n) : 소수점을 n번째 까지만 표현하고, 반올림하고 싶을때
tmp['REG_YYMM'] = tmp['year'] * 100 + tmp['month']
tmp = tmp[['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM', 'AMT']]
tmp = tmp.groupby(['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM']).sum().reset_index(drop = False)

# 디코딩 : 바이트 열을 유니코드로 변환 하는 것

tmp['CARD_SIDO_NM'] = encoders['CARD_SIDO_NM'].inverse_transform(tmp['CARD_SIDO_NM'])
tmp['STD_CLSS_NM'] = encoders['STD_CLSS_NM'].inverse_transform(tmp['STD_CLSS_NM'])

# inverse_transform :  인덱스(숫자)를 입력하면,원본 값을 구할 수 있다.
# inverse_transform :  숫자 --> 문자열로 바꾸는 것

# 제출 파일 만들기
submission = pd.read_csv(path + 'submission.csv', index_col = 0)
submission = submission.drop(['AMT'], axis = 1)
submission = submission.merge(tmp, left_on = ['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM'],
                                   right_on = ['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM'], how = 'left')
submission.index.name = 'id'
submission.to_csv(path + 'mysubmission_200629(3).csv', encoding = 'utf-8-sig')
print(submission.head())