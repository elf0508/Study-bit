# 웹 페이지의 내용을 가져오는 간단한 웹 크롤러. 

# 시작하기 전에 requests와 beautifulsoup4 패키지를 설치.
# pip install requests beautifulsoup4

# 1. 웹 문서 전체 가져오기

from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen("http://www.naver.com")
bsObject = BeautifulSoup(html, "html.parser")

# print(bsObject)

# 2. 타이틀 가져오기
# 태그로 구성된 트리에서 title 태그만 출력. 

# print(bsObject.head.title)      # <title>NAVER</title>

# 3. 모든 메타 데이터의 내용 가져오기
# 웹문서에서 메타 데이터만 찾아서 content 속성값을 가져온다.

# for meta in bsObject.head.find_all('meta'):
#     print(meta.get('content'))

# 4. 원하는 태그의 내용 가져오기 
# find를 사용하면 원하는 태그의 정보만 가져올 수 있다. 

# 예를 들어 www.python.org/about 에서 다음 태그의 content 속성값을 가져오려면..
# <meta content="The official home of the Python Programming Language" name="description"/>


# 우선 웹문서에 있는  meta 태그 중 가져올 태그를 name 속성 값이 description인 것으로 한정한다.

# print(bsObject.head.find("meta", {"name" : "description"}))


# meta 태그의 content 내용을 가져온다. 
# print(bsObject.head.find("meta", {"mane" : "description"}).get('content'))

# 5. 모든 링크의 텍스트와 주소 가져오기
# a 태그로 둘러싸인 텍스트와 a 태그의 href 속성을 출력.

for link in bsObject.find_all('a'):
    print(link.text.strip(), link.get('href'))
