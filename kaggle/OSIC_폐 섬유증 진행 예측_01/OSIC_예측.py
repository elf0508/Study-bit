
import numpy as np
import pandas as pd
import os

from glob import glob     #  특정 파일 리스트 가져오기
from PIL import Image
import cv2

 #  데이터 시각화 모듈 

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns    

from bokeh.plotting import figure  
from bokeh.io import output_notebook, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Panel
from bokeh.models.widgets import Tabs

import albumentations as albu

import math

 # 데이터의 성격을 파악하는 과정 :  EDA(Exploratory Data Analysis, 탐색적 데이터 분석)
 # 단 한 줄의 명령으로 탐색하는 패키지인 판다스 프로파일링(pandas-profiling)
import pandas_profiling   

import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

# 원핫인코딩
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# 이미지 트레이닝 및 테스트 경로 설정

DATASET = '../input/osic-pulmonary-fibrosis-progression'
TEST_DIR = 'test.csv'
TRAIN_CSV_PATH = 'train.csv'

# 테스트 이미지 목록 가져오기

train_fns = glob(DATASET + '*')
test_fns = glob(TEST_DIR + '*')

# 데이터 가져오기

train = pd.read_csv('kaggle/OSIC_폐 섬유증 진행/train.csv')
test = pd.read_csv('kaggle/OSIC_폐 섬유증 진행/test.csv')

# print(train)

#                 FVC : 노력성 폐활량 / 최대로 숨을 들이쉰 다음, 최대 노력으로 끝까지 내쉬었을 때 공기량

'''
             Patient               Weeks  FVC   Percent  Age        Sex SmokingStatus

0     ID00007637202177411956430     -4  2315  58.253649   79  Male     Ex-smoker
1     ID00007637202177411956430      5  2214  55.712129   79  Male     Ex-smoker
2     ID00007637202177411956430      7  2061  51.862104   79  Male     Ex-smoker
3     ID00007637202177411956430      9  2144  53.950679   79  Male     Ex-smoker
4     ID00007637202177411956430     11  2069  52.063412   79  Male     Ex-smoker
...                         ...    ...   ...        ...  ...   ...           ...
1544  ID00426637202313170790466     13  2712  66.594637   73  Male  Never smoked
1545  ID00426637202313170790466     19  2978  73.126412   73  Male  Never smoked
1546  ID00426637202313170790466     31  2908  71.407524   73  Male  Never smoked
1547  ID00426637202313170790466     43  2975  73.052745   73  Male  Never smoked
1548  ID00426637202313170790466     59  2774  68.117081   73  Male  Never smoked

[1549 rows x 7 columns]

'''

sample = pd.read_csv('kaggle/OSIC_폐 섬유증 진행/sample_submission.csv')

# print(sample.head(3))

'''
        Patient_Week                FVC     Confidence

0  ID00419637202311204720264_-12  2000         100
1  ID00421637202311550012437_-12  2000         100
2  ID00422637202311677017371_-12  2000         100

'''


# 교육 데이터의 흡연 상태

# print(train['SmokingStatus'].drop_duplicates())

'''
0             Ex-smoker
36         Never smoked
288    Currently smokes

Name: SmokingStatus, dtype: object

'''

# 흡연 여부의 세 가지 유형

# print(train['Sex'].drop_duplicates())

'''

0       Male
62    Female

Name: Sex, dtype: object

'''
# 성별 가치관이 남성과 여성이고, 다른 가치관은 없다.

# print(train.head(10))

# 결측치 확인

# print(train.isnull().sum())

'''
Patient          0
Weeks            0
FVC              0
Percent          0
Age              0
Sex              0
SmokingStatus    0

dtype: int64

'''

# 주별 최대값과 최소값

print("Minimum number of value for Weeks is: {}".format(train['Weeks'].min()), "\n" +
      "Maximum number of value for Weeks is: {}".format(train['Weeks'].max() ))

# 최소값: -5
# 최대값: 133

# 환자 통계 확인

# print(train['Patient'].describe())

'''
count                          1549
unique                          176
top       ID00167637202237397919352
freq                             10

Name: Patient, dtype: object

'''

# FVC 통계

# print(train['FVC'].describe(percentiles=[0.1,0.2,0.5,0.75,0.9]))

'''
count    1549.000000
mean     2690.479019
std       832.770959
min       827.000000
10%      1650.800000
20%      1997.000000
50%      2641.000000
75%      3171.000000
90%      3874.400000
max      6399.000000

Name: FVC, dtype: float64

'''
# 출력 결과에서 FVC가 2109보다 작으면, 하위 25%에 있다고 말할 수 있다.

# 연령 관련 통계

# print(train['Age'].describe())

'''
count    1549.000000
mean       67.188509
std         7.057395
min        49.000000
25%        63.000000
50%        68.000000
75%        72.000000
max        88.000000

Name: Age, dtype: float64

'''
# 이 집단의 평균 연령이 67세라는 점에서, 이 집단은 주로 고령층인 것으로 보여진다.

# 테스트 데이터
# print(test)

'''
        Patient                Weeks   FVC   Percent  Age   Sex     SmokingStatus

0  ID00419637202311204720264      6  3020  70.186855   73  Male     Ex-smoker
1  ID00421637202311550012437     15  2739  82.045291   68  Male     Ex-smoker
2  ID00422637202311677017371      6  1930  76.672493   73  Male     Ex-smoker
3  ID00423637202312137826377     17  3294  79.258903   72  Male     Ex-smoker
4  ID00426637202313170790466      0  2925  71.824968   73  Male  Never smoked

'''

# 환자 ID 및 Week columns

test_patient_weeklist = test['Patient_Week'] = test['Patient'].astype(str)+"_"+test['Weeks'].astype(str)
test2 = test.drop('Patient', axis=1)
test3 = test.drop('Weeks', axis=1)
test4 = test.reindex(columns=['Patient_Week', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus'])

# print(test4.head(7))

'''
            Patient_Week          FVC    Percent  Age   Sex     SmokingStatus

0   ID00419637202311204720264_6  3020  70.186855   73  Male     Ex-smoker
1  ID00421637202311550012437_15  2739  82.045291   68  Male     Ex-smoker
2   ID00422637202311677017371_6  1930  76.672493   73  Male     Ex-smoker
3  ID00423637202312137826377_17  3294  79.258903   72  Male     Ex-smoker
4   ID00426637202313170790466_0  2925  71.824968   73  Male  Never smoked

'''

# 고유한 환자 ID 수

n = train['Patient'].nunique()
# print(n)        # 176

# 그래프

k = 1 + math.log2(n)

# training data (FVC)

sns.distplot(train['FVC'], kde=True, rug=False, bins=int(k)) 

# 그래프 제목
plt.title('FVC')

plt.show()

# 나이

sns.distplot(train['Age'], kde=True, rug=False, bins=int(k)) 

plt.title('Age')

plt.show()

# 나이와 FVC 간의 상관 관계

sns.scatterplot(data=train, x='Age', y='FVC')

plt.show()

# 나이와 FVC 사이의 상관 계수

df = train
df.corr()['Age']['FVC']

# print(df)

# 흡연자로 좁혀 나이와 FVC 사이의 상관 계수 

df_smk = train.query('SmokingStatus == "Currently smokes"')
df_smk.corr()['Age']['FVC']

# print(df_smk)

# 흡연자, 나이별 산점도 및 FVC

sns.scatterplot(data=df_smk, x='Age', y='FVC')

plt.show()

# 나이와 FVC 간의 상관 관계 

sns.scatterplot(data=train, x='Percent', y='FVC')

plt.show()

# 흡연자에게 초점을 맞출 때, 나이와 fvc 사이에는 아무런 상관관계가 없어 보인다.

# '나이별'로 집계된 FVC에 대한 요약 통계 계산

df.groupby('Age').describe()['FVC']

# print(df.groupby)

# 환자 ID별로 집계된 FVC에 대한 요약 통계량

df.groupby('Patient').describe(percentiles=[0.1,0.2,0.5,0.8])['FVC']

# print(df.groupby('Patient'))

# 상관 관계 개요

df_corr = df.corr()

print(df_corr)

# 상관관계 열 지도 

corr_mat = df.corr(method='pearson')

sns.heatmap(corr_mat,
            vmin=-1.0,
            vmax=1.0,
            center=0,
            annot=True, # True : 그리드에 값 표시
            fmt='.1f',
            xticklabels=corr_mat.columns.values,
            yticklabels=corr_mat.columns.values
           )

plt.show()

# '성별'에 대한 파이 차트

plt.pie(train["Sex"].value_counts(),labels=["Male","Female"],autopct="%.1f%%")

plt.title("Ratio of Sex")

# plt.show()

# 출력 결과를 보아 압도적으로 '남성'임을 알수있다.

# 흡연 상태에 대한 파이 차트

plt.pie(train["SmokingStatus"].value_counts(),labels=["Ex-smoker","Never smoked","Currently smokes"],autopct="%.1f%%")
plt.title("SmokingStatus")

plt.show()

# 출력 결과를 보아 현재 흡연하는 사람들이 훨씬 더 적다는 것을 알 수 있다.

# 전체적인 자료를 보는 것에서 벗어나, 최악의 증상을 보이는 환자들을 집중적으로 찾아보자.

# 영상 데이터의 상위 10%와 하위 10%를 표시

print(train[train.FVC < 1651])


# print(train[train.FVC > 3874])

# training data
train_x = train.drop(['FVC'], axis=1)
train_y = df['FVC']

# 현재값 확인

# print(train_x)

train_x['Patient_Week'] = train_x['Patient'].astype(str)+"_"+train_x['Weeks'].astype(str)

# print(train_x.head(5))

# 범주 변수를 임의 값으로 변환

train_x['Sex'] = train_x['Sex'].map({'Male': 0, 'Female': 1})
train_x['SmokingStatus'] = train_x['SmokingStatus'].map({'Never smoked': 0, 'Ex-smoker': 1, 'Currently smokes': 2})

# 환자 ID 및  Week columns

train_x_patient_weeklist = train_x['Patient_Week'] = train_x['Patient'].astype(str)+"_"+train_x['Weeks'].astype(str)
train_x2 = train_x.drop('Patient', axis=1)
train_x3 = train_x.drop('Weeks', axis=1)
train_x4 = train_x.reindex(columns=['Patient_Week', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus'])

print(train_x4.head(7))

# 변환된 값 확인

print(train_x4)

# 시험 데이터는 특징만 있으니 그냥 복사

test_x = test.copy()

osic_features = ['Percent', 'Age', 'Sex', 'SmokingStatus']

X = train_x4[osic_features]

# 모델

osic_model = DecisionTreeRegressor(random_state=1)

# Fit model

osic_model.fit(X, train_y)

print(X.head())

print("The predictions are")

print(osic_model.predict(X.head()))

# 교육 데이터의 FVC

plt.figure(figsize=(18,6))

plt.plot(train_x4["FVC"], label = "Train_Data")

plt.legend()

# plt.show()

# FVC 예측 시각화

plt.figure(figsize=(18,6))

Y_train_Graph = pd.DataFrame(X)

plt.plot(Y_train_Graph, label = "Predict")

plt.legend()

plt.show()

#  submission file

submission = pd.DataFrame(columns = ["Patient_Week", "FVC", "Confidence"])

submission.to_csv('submission.csv', index=False)