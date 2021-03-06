from flask import Flask

# 모든 Flask 어플리케이션은 어플리케이션 인스턴스(Application Instance)를 생성해야한다.
# 웹 서버는 클라이언트로부터 수신한 모든 리퀴스트를 이 오브젝트에서 처리하는데 
# 이 때 웹 서버 게이트 웨이 인터페이스(WSGI)라는 프로토콜을 사용한다. 
# 이 어플리케이션은 아래 코드와 같이 생성시킨다.


app = Flask(__name__)

# Flask 생성자에 필요한 한 가지 매개변수는 메인 모듈의 이름이나 어플리케이션의 패키지 이름입니다.


# 라우트(Route) 뷰(View)

# 웹 브라우저와 같은 클라이언트는 웹 서버에 리퀘스트를 전송하여 
# 플라스크 어플리케이션 인스턴스에 교대로 전송한다. 
# 어플리케이션 인스턴스는 각 URL 리퀴세트 실행을 위해 어떤 코드가 필요한지 알아야 하며, 
# 따라서 URL을 파이썬 함수에 매핑시켜야 하는데 이 URL을 처리하는 함수를 라우트라고 한다.

# 플라스크 에서는 아래와 같이 데코레이터를 이용하여 사용합니다.


@app.route('/')
def index():
    return '<h1> Hello Flask!! </h1>'   #  127.0.0.1:5000 으로 접속 성공시, 보이는 글씨

# 그리고 index()와 같은 함수를 뷰 함수라고 한다.

# @app.route('/Flask/<name>')
# def index():
#     return '<h1> Hello, %s!! </h1>' %name


# 서버 시작

if __name__ == '__main__':
    app.run()


