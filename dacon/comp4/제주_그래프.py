import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl1
from matplotlib import rc

import warnings
warnings.filterwarnings(action = 'ignore')

# %config InlineBackend.figure_format = 'retina'
 
# !apt -qq -y install fonts-nanum
 
# 한글 폰트 설정

mpl.rcParams['axes.unicode_minus'] = False

# 폰트 경로
font_path = "C:/Windows/Fonts/malgunbd.ttf"

#폰트 이름 얻어오기
font_name = fm.FontProperties(fname=font_path).get_name()

#font 설정
mpl.rc('font',family=font_name)

mpl.font_manager.get_fontconfig_fonts()

# # print ('버전: ', mpl.__version__)
# # print ('설치 위치: ', mpl.__file__)
# # print ('설정 위치: ', mpl.get_configdir())
# # print ('캐시 위치: ', mpl.get_cachedir())

# # 버전:  3.1.3
# # 설치 위치:  C:\Users\bitcamp\anaconda3\lib\site-packages\matplotlib\__init__.py     
# # 설정 위치:  C:\Users\bitcamp\.matplotlib  
# # 캐시 위치:  C:\Users\bitcamp\.matplotlib  

# # print ('설정파일 위치: ', mpl.matplotlib_fname())
# # 설정파일 위치:  C:\Users\bitcamp\anaconda3\lib\site-packages\matplotlib\mpl-data\matplotlibrc

# font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')

# # ttf 폰트 전체갯수
# print(len(font_list))  # 152

# OSX 의 설치 된 폰트를 가져오는 함수
font_list_mac = fm.OSXInstalledFonts()
print(len(font_list_mac))

# # 시스템 폰트에서 읽어온 리스트에서 상위 10개만 출력
# font_list[:10] 

# f = [f.name for f in fm.fontManager.ttflist]
# print(len(font_list))

# # 10개의 폰트명 만 출력
# f[:10]

# #  나눔고딕을 사용할 예정이기 때문에 이름에 ‘Nanum’이 들어간 폰트만 가져와 봅니다.

[(f.name, f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name]

# fig, ax = plt.subplots()
# ax.plot(10*np.random.randn(100), 10*np.random.randn(100), 'o')
# ax.set_title('숫자 분포도 보기')
# plt.show()

fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
# fontpath = 'D:/Study-bit/dacon/comp4/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)
plt.rc('font', family='NanumBarunGothic') 
mpl.font_manager._rebuild()

plt.figure(figsize=(5,5))
plt.plot([0,1], [0,1], label='한글테스트용')
plt.legend()
# plt.show()

data = pd.read_csv('D:/Study-bit/dacon/comp4/201901-202003.csv')

sub = pd.read_csv('D:/Study-bit/dacon/comp4/submission.csv')

data.head()

print("data.head() : ", data.head())

# data.head() :     REG_YYMM CARD_SIDO_NM  ...      AMT CNT
# 0    201901           강원  ...   311200  
#  4
# 1    201901           강원  ...  1374500  
#  8
# 2    201901           강원  ...   818700  
#  6
# 3    201901           강원  ...  1717000  
#  5
# 4    201901           강원  ...  1047300  
#  3

# [5 rows x 12 columns]

# data[data['CNT']==data['CNT'].max()]

# data.shape

# print("data.shape : ", data.shape)  # (24697792, 12)

# data.isna().sum()

# print("data.isna().sum() : ", data.isna().sum())

# data.isna().sum() :  REG_YYMM
# 0
# CARD_SIDO_NM         0
# CARD_CCG_NM      87213
# STD_CLSS_NM          0
# HOM_SIDO_NM          0
# HOM_CCG_NM      147787
# AGE                  0
# SEX_CTGO_CD          0
# FLC                  0
# CSTMR_CNT            0
# AMT                  0
# CNT                  0

data['REG_YYMM'].unique()

print("data['REG_YYMM'].unique() : ", data['REG_YYMM'].unique())

#  [201901 201902 201903 201904 201905 201906 201907 201908 201909 201910
#  201911 201912 202001 202002 202003] 

sub.head()

print("sub.head() : ", sub.head())

# id  ...           AMT    
# 0   0  ...  9.605901e+07
# 1   1  ...  2.915798e+09
# 2   2  ...  9.948169e+08
# 3   3  ...  1.331730e+07
# 4   4  ...  0.000000e+00

# [5 rows x 5 columns]

sub.groupby('CARD_SIDO_NM')['STD_CLSS_NM'].unique()

# 지역 구성
data['CARD_SIDO_NM'].unique()

# 시,군,구 구성
data['CARD_CCG_NM'].unique()

# 업종 구성
data['STD_CLSS_NM'].unique()

##### EDA ####

# value_count를 활용한 지역별 업종 비중 확인
# 각 업종별로 이용 건수, 고객수 비율이 높은 지역을 뽑자.
# 등장 횟수 EDA (지역별 업종 비중)

# 업종별 등장 빈도수
# 막대 그래프

fig = plt.figure(figsize=(20, 15))
fig.patch.set_facecolor('xkcd:mint green')
sns.barplot(y=data['STD_CLSS_NM'].value_counts().index,x=data['STD_CLSS_NM'].value_counts())
plt.tight_layout()
# plt.show()

city_count= data.groupby(['CARD_SIDO_NM','CARD_CCG_NM'])['STD_CLSS_NM'].value_counts().reset_index(name='count')
city_count.head()

# 사실상 시,군,구 변수는 활용을 할 수 없기때문에 도별로 접근하는 것이 좋을 것으로 보인다.
city_sum = city_count.groupby(['CARD_SIDO_NM','STD_CLSS_NM'])['count'].sum().reset_index(name='sum')
city_sum.head()

# 도, 특별시, 광역시에 따른 업종별 소비 등장 횟수
# 동그라미 그래프

fig,axs = plt.subplots(4, 4)
fig.set_size_inches(30,20)
fig.patch.set_facecolor('xkcd:mint green')

for idx,city in enumerate(city_sum['CARD_SIDO_NM'].unique()):
  if idx <=3:
    axs[0, idx].pie(city_sum[city_sum['CARD_SIDO_NM']==city].sort_values('sum',ascending=False).head(10)['sum'],autopct='%.0f%%',
                  labels= city_sum[city_sum['CARD_SIDO_NM']==city].sort_values('sum',ascending=False).head(10)['STD_CLSS_NM'],
                  radius= 1.2,
                  startangle=90,
                  counterclock=False)
    axs[0, idx].title.set_text(city)
  if idx > 3 and idx <=7:
    axs[1, idx-4].pie(city_sum[city_sum['CARD_SIDO_NM']==city].sort_values('sum',ascending=False).head(10)['sum'],autopct='%.0f%%',
                  labels= city_sum[city_sum['CARD_SIDO_NM']==city].sort_values('sum',ascending=False).head(10)['STD_CLSS_NM'],
                  radius= 1.2,
                  startangle=90,
                  counterclock=False)
    axs[1, idx-4].title.set_text(city)
  if idx > 7 and idx <= 11:
    axs[2, idx-8].pie(city_sum[city_sum['CARD_SIDO_NM']==city].sort_values('sum',ascending=False).head(10)['sum'],autopct='%.0f%%',
                  labels= city_sum[city_sum['CARD_SIDO_NM']==city].sort_values('sum',ascending=False).head(10)['STD_CLSS_NM'],
                  radius= 1.2,
                  startangle=90,
                  counterclock=False)
    axs[2, idx-8].title.set_text(city)
  if idx > 11 and idx <=15:
    axs[3, idx-12].pie(city_sum[city_sum['CARD_SIDO_NM']==city].sort_values('sum',ascending=False).head(10)['sum'],autopct='%.0f%%',
                  labels= city_sum[city_sum['CARD_SIDO_NM']==city].sort_values('sum',ascending=False).head(10)['STD_CLSS_NM'],
                  radius= 1.2,
                  startangle=90,
                  counterclock=False)
    axs[3, idx-12].title.set_text(city)

#plt.tight_layout()
# plt.show()

# SIDO = city_count[city_count['CARD_SIDO_NM']=='충북']

# fig,axs = plt.subplots(4, 4)
# fig.set_size_inches(40,30)
# fig.patch.set_facecolor('xkcd:mint green')

# for idx,city in enumerate(SIDO['CARD_CCG_NM'].unique()):
#   if idx <=3:
#     sns.barplot(y=SIDO[SIDO['CARD_CCG_NM']==city].sort_values('count',ascending=False).head(10)['STD_CLSS_NM'],
#                 x=SIDO[SIDO['CARD_CCG_NM']==city].sort_values('count',ascending=False).head(10)['count'],ax = axs[0][idx])
#     axs[0, idx].title.set_text(city)
#   if idx > 3 and idx <=7:
#     sns.barplot(y=SIDO[SIDO['CARD_CCG_NM']==city].sort_values('count',ascending=False).head(10)['STD_CLSS_NM'],
#                 x=SIDO[SIDO['CARD_CCG_NM']==city].sort_values('count',ascending=False).head(10)['count'],ax = axs[1][idx-4])
#     axs[1, idx-4].title.set_text(city)
#   if idx > 7 and idx <= 11:
#     sns.barplot(y=SIDO[SIDO['CARD_CCG_NM']==city].sort_values('count',ascending=False).head(10)['STD_CLSS_NM'],
#                 x=SIDO[SIDO['CARD_CCG_NM']==city].sort_values('count',ascending=False).head(10)['count'],ax = axs[2][idx-8])
#     axs[2, idx-8].title.set_text(city)
#   if idx > 11 and idx <=15:
#     sns.barplot(y=SIDO[SIDO['CARD_CCG_NM']==city].sort_values('count',ascending=False).head(10)['STD_CLSS_NM'],
#                 x=SIDO[SIDO['CARD_CCG_NM']==city].sort_values('count',ascending=False).head(10)['count'],ax = axs[3][idx-12])
#     axs[3, idx-12].title.set_text(city)

# #plt.tight_layout()
# plt.show()

'''경기도 같은 경우 42개의 도시가 잡힌다. 그러므로 시,군,구 별로 한꺼번에 그래프로 비교하기 힘들 것으로 보임.'''

# 이용 건수, 고객수 비율
# 업종별 이용 건수

print("data : ", data)  # [24697792 rows x 12 columns]

stuff = data.groupby('STD_CLSS_NM')[['CNT','CSTMR_CNT']].sum().reset_index()
stuff.head()

def make_bar(data,x_col,y_col):
    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor('xkcd:mint green')
    sns.barplot(x=x_col,y=y_col,data=data.sort_values(x_col,ascending=False))
    plt.title(x_col)
#   plt.show()

# 이용고객수
make_bar(stuff,'CSTMR_CNT','STD_CLSS_NM')

# 이용 횟수
make_bar(stuff,'CNT','STD_CLSS_NM')

# 1,2 위를 주목해볼 필요가 있는 것으로 보인다. 이용 고객수와 이용 횟수의 순위의 1,2위가 바뀐다.

# 이용고객수에 비해 이용건수의 크기가 작은 경우 -> 카드 취소가 많은 업종
# 이용고객수에 비해 이용건수가 큰 경우 -> 같은 고객이 많이 오는 업종

data['gap']= data['CNT'] - data['CSTMR_CNT']
data[data['gap'] == -236]

# 20년 2월의 항공업같은 경우, 40대 여성(가구생애주기 =3)의 카도 취소량이 236건이나 된다.

# gap < 0 : 카드 취소 O
# gap > 0 : 같은 이용고객들 O

data.loc[data['gap'] <0,'mark'] = '취소있음'
data.loc[data['gap'] ==0,'mark'] = '고객다름'
data.loc[data['gap'] >0,'mark'] = '단골있음'

gap=data.groupby('STD_CLSS_NM')['mark'].value_counts().reset_index(name='count')
gap.head()

# 카드 취소 내역이 있는 업종

df=gap.groupby('STD_CLSS_NM')['count'].sum().reset_index().merge(gap[gap['mark']=='취소있음'][['STD_CLSS_NM','count']],on='STD_CLSS_NM')
df.rename(columns={'count_x': 'total',
                   'count_y': 'cancel_count'},inplace=True)
df['rate'] = df['cancel_count']/df['total']

df=df.sort_values('rate',ascending=False,ignore_index=True)
df.head()

# 항공 운송업과 여행사업이 카드취소 고객의 존재가 독보적으로 높다.

# 새로운 고객으로만 구성된 업종

df=gap.groupby('STD_CLSS_NM')['count'].sum().reset_index().merge(gap[gap['mark']=='고객다름'][['STD_CLSS_NM','count']],on='STD_CLSS_NM')
df.rename(columns={'count_x': 'total',
                   'count_y': 'differ_count'},inplace=True)
df['rate'] = df['differ_count']/df['total']

df=df.sort_values('rate',ascending=False,ignore_index=True)
df.head()

# 랜트카, 여행사업, 관광사업

# 새로운 고객들의 유입이 많은 업종들은 대부분 관광업종으로 구성되어 있는 것으로 보인다.
# 관광업종의 비증이 높은 지역을 찾아보는 것이 좋을 것 같음.

# 단골 고객이 있는 업종

df=gap.groupby('STD_CLSS_NM')['count'].sum().reset_index().merge(gap[gap['mark']=='단골있음'][['STD_CLSS_NM','count']],on='STD_CLSS_NM')
df.rename(columns={'count_x': 'total',
                   'count_y': 'differ_count'},inplace=True)
df['rate'] = df['differ_count']/df['total']

df=df.sort_values('rate',ascending=False,ignore_index=True)
df.head(20)

# 우리가 일상생활에서 많이 접하는 업종들이 대부분 같은 고객이 찾아오는 경우가 많은 것으로 확인됨.

# 위와 같은 방식으로 월별, 지역별로 CNT, CSTMR_CNT를 확인해 볼 수도 있을 것으로 보인다.

# 또한 매월 업종별, 각 지역에 따른 업종별 -> CNT, CSTMR_CNT를 확인해봐도 괜찮을 것으로 보임.

# 결과적으로 CNT와 CSTMR_CNT를 활용해 AMT를 예측할 수 있는 방법을 도출하는 것이 좋을 것 같음.
# 지역별 이용 건수

data.drop(['gap','mark'],axis=1,inplace=True) # 데이터 원상복구
data.head()

df = data.groupby('CARD_SIDO_NM')[['CSTMR_CNT','CNT']].sum().reset_index()
df.head()
df['CSTMR_CNT']/df['CNT']

# 이걸로만 봤을 때 새로운 고객의 유입과 단골 손님의 유무를 제대로 파악하기 어려워 보임.

# 지역에 따른 업종별 이용건수

df = data.groupby(['CARD_SIDO_NM','STD_CLSS_NM'])[['CSTMR_CNT','CNT']].sum().reset_index()
df.head()
df[df['CSTMR_CNT']>df['CNT']]

# 제주도의 항공 운송업과 광주, 대구, 대전의 카드승인 취소가 어느정도 나타나는 것으로 보인다.

# 월별로 이용고객의 추이를 확인할 필요가 있을 것으로 보임.
# 재방문율 변수 만들기

df['re_visit']=1 - df['CSTMR_CNT']/df['CNT']
df[df['CARD_SIDO_NM'] == '강원'].sort_values('re_visit',ascending=False).head()

'''예상대로 편의점이 재방문율이 높게 나옴'''

fig,axs = plt.subplots(4, 4)
fig.set_size_inches(30,20)
fig.patch.set_facecolor('xkcd:mint green')

for idx,city in enumerate(df['CARD_SIDO_NM'].unique()):
  if idx <=3:
    axs[0, idx].pie(df[df['CARD_SIDO_NM']==city].sort_values('re_visit',ascending=False).head(10)['re_visit'],autopct='%.0f%%',
                  labels= df[df['CARD_SIDO_NM']==city].sort_values('re_visit',ascending=False).head(10)['STD_CLSS_NM'],
                  radius= 1.2,
                  startangle=90,
                  counterclock=False)
    axs[0, idx].title.set_text(city)
  if idx > 3 and idx <=7:
    axs[1, idx-4].pie(df[df['CARD_SIDO_NM']==city].sort_values('re_visit',ascending=False).head(10)['re_visit'],autopct='%.0f%%',
                  labels= df[df['CARD_SIDO_NM']==city].sort_values('re_visit',ascending=False).head(10)['STD_CLSS_NM'],
                  radius= 1.2,
                  startangle=90,
                  counterclock=False)
    axs[1, idx-4].title.set_text(city)
  if idx > 7 and idx <= 11:
    axs[2, idx-8].pie(df[df['CARD_SIDO_NM']==city].sort_values('re_visit',ascending=False).head(10)['re_visit'],autopct='%.0f%%',
                  labels= df[df['CARD_SIDO_NM']==city].sort_values('re_visit',ascending=False).head(10)['STD_CLSS_NM'],
                  radius= 1.2,
                  startangle=90,
                  counterclock=False)
    axs[2, idx-8].title.set_text(city)
  if idx > 11 and idx <=15:
    axs[3, idx-12].pie(df[df['CARD_SIDO_NM']==city].sort_values('re_visit',ascending=False).head(10)['re_visit'],autopct='%.0f%%',
                  labels= df[df['CARD_SIDO_NM']==city].sort_values('re_visit',ascending=False).head(10)['STD_CLSS_NM'],
                  radius= 1.2,
                  startangle=90,
                  counterclock=False)
    axs[3, idx-12].title.set_text(city)

#plt.tight_layout()
# plt.show()

# 대부분의 지역 모두 슈퍼마켓, 편의점에 대한 재방문율이 높게 나타나는 것으로 보이지만 지역마다 어느정도 특색이 보일만한 업종이 보인다.

# 전북의 경우 '내항 여객 운송업'이 1위이다.
# 관광업의 특색이 있는 지역(휴양콘도)
# 유일하게 보이는 강원도의 '욕탕업'
# 날짜별 이용 건수

df = data.groupby('REG_YYMM')[['CSTMR_CNT','CNT']].sum().reset_index()
#df['REG_YYMM']=df['REG_YYMM'].astype('object')
df.head()

make_bar(df,'REG_YYMM','CSTMR_CNT')

make_bar(df,'REG_YYMM','CNT')

# 2020년 1월을 기점으로 이용고객수와 카드이용횟수가 줄어드는 것으로 보이지만 전체적으로 봤을 때 뚜렷한 특징을 찾기는 힘들다.

# 19년의 데이터가 20년도를 예측하는데 도움이 될 수 있을지 의문임.
# 날짜에 따른 업종별 이용건수

df = data.groupby(['REG_YYMM','STD_CLSS_NM'])[['CSTMR_CNT','CNT']].sum().reset_index()
df.head()

# 각 날짜마다 고객수와 이용횟수가 높은 업종을 뽑아보자.

# 날짜 변경 -> 일반 숫자로
for idx,date in enumerate(df['REG_YYMM'].unique()):
  df.loc[df['REG_YYMM']==date,'new_date'] = idx

fig = plt.figure(figsize=(5, 3))
fig.patch.set_facecolor('xkcd:mint green')
sns.lineplot(x='new_date',y='CSTMR_CNT',data=df[df['STD_CLSS_NM']=='건강보조식품 소매업'],hue='STD_CLSS_NM')
# plt.show()

def make_line(data,x_col,y_1,y_2,hue_col):
    fig,(ax1,ax2) = plt.subplots(ncols=2)
    #print(axs)
    fig.set_size_inches(10,5)
    fig.patch.set_facecolor('xkcd:mint green')
    sns.lineplot(x=x_col,y=y_1,data=df[df['STD_CLSS_NM']== hue_col],hue='STD_CLSS_NM',ax=ax1)
    fig.patch.set_facecolor('xkcd:mint green')
    sns.lineplot(x=x_col,y=y_2,data=df[df['STD_CLSS_NM']== hue_col],hue='STD_CLSS_NM',ax=ax2)
    # plt.show()

for hue_col in df['STD_CLSS_NM'].unique():
      make_line(df,'new_date','CSTMR_CNT','CNT',hue_col)

# 코로나가 발생한 이후로 대부분 업종의 고객이 줄어들었지만 약간의 반등을 보이는 업종들도 몇개 보인다.

# 앞으로 맞춰야할 AMT와의 비교를 통해 CMT를 활용하는 방법을 생각하면 좋을 것으로 보인다.
# 날짜에 따른 업종별 이용건수와 AMT의 변화

df = data.groupby(['REG_YYMM','STD_CLSS_NM'])[['CSTMR_CNT','CNT','AMT']].sum().reset_index()
df.head()

# 날짜 변경 -> 일반 숫자로
for idx,date in enumerate(df['REG_YYMM'].unique()):
  df.loc[df['REG_YYMM']==date,'new_date'] = idx

df[['CSTMR_CNT','CNT','AMT']]  # 615 rows × 3 columns

def make_line(data,x_col,y_1,y_2,y_3,hue_col):
    fig,(ax1,ax2,ax3) = plt.subplots(ncols=3)
    fig.patch.set_facecolor('xkcd:mint green')
    fig.set_size_inches(10,5)
  
    sns.lineplot(x=x_col,y=y_1,data=df[df['STD_CLSS_NM']== hue_col],hue='STD_CLSS_NM',ax=ax1)
    sns.lineplot(x=x_col,y=y_2,data=df[df['STD_CLSS_NM']== hue_col],hue='STD_CLSS_NM',ax=ax2)
    sns.lineplot(x=x_col,y=y_3,data=df[df['STD_CLSS_NM']== hue_col],hue='STD_CLSS_NM',ax=ax3)
    # plt.show()


# for hue_col in df['STD_CLSS_NM'].unique():
#   make_line(df,'new_date','CSTMR_CNT','CNT','AMT',hue_col)

'''메모리가 커서 그래프 그리는 것은 주석처리함'''
# 사실상 CNT 와 CSTMR_CNT가 AMT를 반영한다고 해도 무방할 것으로 상당히 흡사한 모양을 띄고 있음.

# AMT에 대한 변수를 찾기 어려울 것 같으면 사실상 CNT와 CSMTR_CNT를 예측할 수 있는 변수를 찾아도 될 것으로 보인다.



























































































