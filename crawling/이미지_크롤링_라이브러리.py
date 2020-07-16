# 이미지 크롤링
# 라이브러리 사용 - 사진 한번에 크롤링 하기
# 설치 : pip install google_images_download

from google_images_download import google_images_download   # importing the library

response = google_images_download.googleimagesdownload()   # class instantiation

arguments = {"keywords":"Polar bears,baloons,Beaches","limit":20,"print_urls":True}  # creating list of arguments
paths = response.download(arguments)   # passing the arguments to the function
print(paths)   # printing absolute paths of the downloaded images

# 키워드랑 리미트만 조절하면 원하는거 가능.

# 또, input argument로 옵션을 살펴서 원하는 기능 가져오기 가능.
# 예) format 으로 원하는 확장자만 다운받을 수 있게.

from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
from urllib.parse import quote_plus

baseUrl = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query='
plusUrl = input('검색어를 입력하세요 : ')
# 한글 검색 자동 변환
url = baseUrl + quote_plus(plusUrl)
html = urlopen(url)
soup = bs(html, "html.parser")
img = soup.find_all(class_='_img')

n = 1
for i in img:
    imgUrl = i['data-source']
    with urlopen(imgUrl) as f:
        with open('D:\Study-bit\crawling\img\dog' + plusUrl + str(n)+'.jpg','wb') as h: # w - write b - binary
            img = f.read()
            h.write(img)
    n += 1
print('다운로드 완료')

soup.find_all("a", limit=2)

# n = 1
# for i in img:
#     imgUrl = i['data-source']
#     with urlopen(imgUrl) as f:
#         with open('D:\Study-bit\crawling\img\cat' + plusUrl + str(n)+'.jpg','wb') as h: # w - write b - binary
#             img = f.read()
#             h.write(img)
#     n += 1
# print('다운로드 완료')

# soup.find_all("a", limit=2)



