import requests
from bs4 import BeautifulSoup

# 한 건의 대화에 대한 정보를 담는 클래스 정의
class Conversation:
    # 질문(Question), 응답(Answer) 두 변수로 구성된다.
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer

    def __str__(self):
        return "질문 : " + self.question + "\n답변 : " + self.answer + "\n"


# 모든 영어 대화 주제를 추출하는 함수
def get_subjects():
    subjects = []

    #전체 주제 목록을 보여주는 페이지로의 요청 객체를 생성
    req = requests.get('https://basicenglishspeaking.com/daily-english-conversation-topics/')
    # 파이썬 크롤러가 접속
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')

    divs = soup.findAll('div', {"class": "tcb-flex-row tcb--cols--3"})
    for div in divs:
        # 내부에 존재하는 <a> 태그들을 추출한다.
        links = div.finfAll('a')

        # <a>태그 내부의 텍스트 리스트에 삽입
        for link in links:
            subject = link.text
            subjects.append(subject)

    return subjects

subjects = get_subjects()

print('총 ', len(subjects), '개의 주제를 찾았습니다.')
print(subjects)

conversations = []
i = 1

# 모든 대화 주제 각각에 접근
for sub in subjects:
    print('(', i, '/', len(subjects), ')', sub)


    # 대화 스트립트를 보여주는 페이지로의 요청 객체 생성
    req = requests.get('http://basicenglishspeaking.com/' + sub)
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')

    qnas = soup.findAll('div', {"class" : "sc_player_container1"})

    # 각각의 대화 내용에 모두 접근한다.
    for qna in qnas:
        if qnas.index(qna) % 2 == 0:
            q = qna.next_sibling
        else:
            a = qna.next_sibling
            c = Conversation(q, a)
            conversations.append(c)

    i = i + 1

    if i == 10:
        break;

print('총', len(conversations), '개의 대화를 찾았습니다.')

for c in conversations:
    print(str(c))


