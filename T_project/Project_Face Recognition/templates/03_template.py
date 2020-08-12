# 템플릿

# 데이터를 연결하여 거대한 테이블 형태로 되어있는 HTML 코드를 구축한다고 생각해보자. 
# 이 데이터는 필요한 HTML 문자열을 사용하여 데이터베이스로부터 얻은 것이다. 
# 프레젠테이션 로직을 템플릿으로 이동시키는 것은 애플리케이션 유지 보수성을 향상시키는데 도움을 준다.
# 템플릿은 응답 텍스트를 포함하고 있는 파일이며, 리퀘스트 내용에서 인식가능한 동적 파트에 대한 변수들을 포함한다. 
# 변수들을 실제 값으로 바꾸는 프로세스와 최종 응답 문자열을 리턴하는 프로세스를 렌더링이라고 한다.


# FlaskProgram/app.py

from flask import Flask, render_template
from flask import url_for,render_template,request

app = Flask(__name__)

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/program/<name>')

def program(name):
    return render_template('program.html', name=name)

if __name__=='__main__':
    app.run()





