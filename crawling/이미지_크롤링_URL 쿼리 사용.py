# URL 쿼리를 사용한 간단한 이미지 크롤링

from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
from urllib.parse import quote_plus
from pathlib import Path

baseUrl = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query='
people = {'puppy': ['강다니엘', '백현', '박보검', '송중기'],
          'cat': ['황민현', '시우민', '이종석', '강동원', '이종석', '이준기'],
          'bear': ['마동석', '조진웅', '조세호', '안재홍'],
          'dinosaur': ['윤두준', '이민기', '육성재', '공유', '김우빈'],
          'rabbit': ['정국', '바비', '박지훈', '수호']}
Path("./img").mkdir(parents=True, exist_ok=True)
for k, v in people.items():
    Path("./img/" + k).mkdir(parents=True, exist_ok=True)
    for person in v:
        url = baseUrl + quote_plus(person)
        html = urlopen(url)
        soup = bs(html, "html.parser")
        img = soup.find_all(class_='_img', limit=50)
        Path("./img/" + k + '/' + person).mkdir(parents=True, exist_ok=True)
        n = 1
        for i in img:
            imgUrl = i['data-source']
            with urlopen(imgUrl) as f:
                with open('./img/' + k + '/' + person + '/' + person + ' ' + str(n)+'.jpg','wb') as h: # w - write b - binary
                    img = f.read()
                    h.write(img)
            n += 1
print('다운로드 완료')