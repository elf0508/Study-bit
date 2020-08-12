from flask import Flask
from flask import request

# 플라스크에는 2 가지 컨텍스트가 있다. 
# 하나는 어플리케이션 컨텍스트(application context), 리퀘스트 컨텍스트(context)가 있다. 
# 플라스크에서는 뷰 함수가 리퀘스트 오브젝트를 접근하는 방법으로 인수를 전송하는 것이 있다. 
# 그러나 이 방법은 어플리케이션의 모든 뷰 함수가 여분의 인수를 갖게 하도록 요구되어 
# 뷰 함수에서 필요 이상의 인수를 갖게된다. 
# 그래서 플라스크에서는 컨텍스트를 사용하여 임시적으로 오브젝트를 글로벌하게 접근할 수 있게한다.


app = Flask(__name__)

@app.route('/')
def Index():
    user = request.headers.get('User')

    return '<h2> %s </h> % user'

# 어플리케이션 컨텍스트

# - current_app : 활성화된 어플리케이션을 위한 인스턴스
# - g : 리퀴스트를 처리하는 동안 어플리케이션이 임시 스토리지를 사용할 수 있는 오브젝트, 이 변수는 각 리퀘스트에 따라 리셋된다.

# 리퀘스트 컨텍스트

# - request : 클라이언트에 의해 송신된 HTTP 리퀘스트의 컨텐츠를 캡슐화하는 오브젝트
# - session : 사용자 세션이며, 어플리케이션이 리퀘스트 사이의 "remembered" 인 값들을 저장하는 데 사용하는 딕셔너리

# 리퀘스트

# 어플리케이션의 클라이언트에서 리퀘스트를 수신하면, 그것을 서비스하기 위해 실행할 뷰 함수가 무엇인지 검색한다. 
# 플라스크에서는 어플리케이션 URL 맵에서 리퀘스트에 주어진 URL을 검토하고 그것을 처리할 뷰 함수에 URL의 매핑을 포함하고 있는지 찾는다. 
# 플라스크에서는 app.route, app.add_url_rule을 사용한다.    

# URL 맵에 있는 HEAD, OPTION, GET 항목은 라우트에 처리되는 리퀘스트 메소드다. 
# HEAD, OPTION은 플라스크에 의해 자동으로 관리된다.

 
# 리퀘스트 후크

# 뷰 함수에서 코드를 중복하여 생성하는 대신 플라스크에는 옵션을 제공하여 공통 함수를 등록하고 
# 리퀘스트가 뷰 함수에 디스패치되는 전후에 실행되도록 하는데, 이 함수를 리퀘스트 후크라 하며 
# 데코레이터를 사용하여 구현한다.


# 플라스크에서는 아래 4개가 제공된다.

# -before_first_request

# -before_request

# -after_request

# -teardown_request


# 응답

# 플라스크에서 HTTP 프로토콜 상태코드를 리턴 값으로 추가할 수 있다. 디폴트 값은 200이다.


app = Flask(__name__)

@app.route('/')

def Index():
    return '<p> Hello Flask </p>', 400

if __name__ == '__main__':
    app.run()

# * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
# 127.0.0.1 - - [20/Jun/2019 15:06:28] "GET / HTTP/1.1" 400 -
# 127.0.0.1 - - [20/Jun/2019 15:06:28] "GET /favicon.ico HTTP/1.1" 404 -


