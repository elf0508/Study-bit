# 실시간 검색 순위 크롤링 & 저장

from bs4 import BeautifulSoup
from urllib.request import urlopen


# with urlopen('https://en.wikipedia.org/wiki/Main_Page') as response:
#     soup = BeautifulSoup(response, 'html.parser')
#     for anchor in soup.find_all('a'):
#         print(anchor.get('href', '/'))

# 위에서 불러운 urlopen으로 링크를 리스폰스로 담겠다. with as 구문 다음처럼 바꾸기 가능.


response = urlopen('https://en.wikipedia.org/wiki/Main_Page')

# beautiful soup 함수 이용해서 리스폰스 넣고 html.parser이용해서 분석한 다음 변수 soup에 담아줌
soup = BeautifulSoup(response, 'html.parser')

# for 문 이용해서 soup에서 모든 a태그 찾아서 변수 anchor 에 넣기
# anchor 를 하나씩 가져와서 주소를 프린트해라.      
for anchor in soup.find_all('a'):
    print(anchor.get('href', '/'))

# 사이트에서 실시간 검색 부분 코드를 보고 그 코드 가져올 수 있는 특징 잡아서 활용
# python
# response = urlopen('사이트 주소')
# soup = BeautifulSoup(response, 'html.parser')
# for anchor in soup.select("스펜.클래스이름")
#     print(anchor)

response = urlopen('https://www.daum.net/')
soup = BeautifulSoup(response, 'html.parser')
for anchor in soup.select("a.link_favorsch"):
    print(anchor)

# response = urlopen('https://www.naver.com/')
# soup = BeautifulSoup(response, 'html.parser')
# for anchor in soup.select("a.link_rel"):
#     print(anchor)


# 결과에서 내용만 뽑아오기
print(soup.get_text())
# print(anchor.get_test())

# 순위추가
i = 1
for anchor in soup.select("a.link_favorsch"):
    print(str(i) + "위: " + anchor.get_text())
    i += 1

# 텍스트 파일로 저장하기
response = urlopen('https://www.daum.net/')
soup = BeautifulSoup(response, 'html.parser')
i = 1
f = open("새파일.txt", 'w')
for anchor in soup.select("a.link_favorsch"):
    data = str(i) + "위: " + anchor.get_text() + "\n"
    i += 1
    f.write(data)
f.close()
