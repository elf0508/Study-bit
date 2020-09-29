# linear algebra
import numpy as np
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
#Unix commands
import os

# import useful tools
from glob import glob
from PIL import Image
import cv2

# import data visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from bokeh.plotting import figure
from bokeh.io import output_notebook, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Panel
from bokeh.models.widgets import Tabs

# import data augmentation
import albumentations as albu

# import math module
import math

#Libraries
import pandas_profiling
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

# One-hot encoding
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

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Display of training data
# print(train)

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

#Loading Sample Files for Submission
sample = pd.read_csv('sample_submission.csv')

# Confirmation of the format of samples for submission
# print(sample.head(3))

'''
        Patient_Week                FVC     Confidence

0  ID00419637202311204720264_-12  2000         100
1  ID00421637202311550012437_-12  2000         100
2  ID00422637202311677017371_-12  2000         100

'''

#Loading Sample Files for Submission
sample = pd.read_csv('sample_submission.csv')

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
# 성별 가치관이 남성과 여성이고 다른 가치관은 없다.

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

# 주별 최대값과 최대값

# print("Minimum number of value for Weeks is: {}".format(train['Weeks'].min()), "\n" +
#       "Maximum number of value for Weeks is: {}".format(train['Weeks'].max() ))

# Minimum number of value for Weeks is: -5
# Maximum number of value for Weeks is: 133

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
# 출력 결과에서 FVC가 2109보다 작으면 하위 25%에 있다고 말할 수 있다.

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
# 이 집단의 평균 연령이 67세라는 점에서 이 집단은 주로 고령층인 것으로 보여진다.

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

# Graph Title
plt.title('FVC')

# Show Histogram
plt.show()

# 나이

sns.distplot(train['Age'], kde=True, rug=False, bins=int(k)) 
# Title of the study data age graph

plt.title('Age')

# Display a histogram of the age of the training data
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
