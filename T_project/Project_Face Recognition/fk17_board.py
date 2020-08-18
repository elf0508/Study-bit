'''
from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

# 터미널 창에서
conn = sqlite3.connect('./data/wanggun.db')
cursor = conn.cursor()
cursor.execute('select *  from general')
print(cursor.fetchall()) # 전체를 출력 하겠다 # (아이디, 이름, 전투력)


@app.route('/') # 서버단에서
def run() :
    conn = sqlite3.connect('./data/wanggun.db') # 접속했다
    c = conn.cursor()
    c.execute('select * from general')
    rows = c.fetchall()
    return render_template('board_index.html', rows=rows)
        # 현재 작업폴더(flask) 하단에 있어야함

@app.route('/modi')
def modi():
    id = request.args.get('id')
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    c.execute('SELECT * FROM general WHERE id = '+str(id))
    rows = c.fetchall()
    return render_template('board_modi.html', rows=rows)


@app.route('/modi') # modi는 html에서 연결해줬던 것
def modi() :
    id = request.args.get('id') # id 요청해놓고
    conn = sqlite3.connect('./data/wanggun.db') # connection 한다
    c = conn.cursor()
    c.execute('select * from general where id = '+str(id))
    rows = c.fetchall()
    return render_template('board_modi.html', rows = rows)

@app.route('/modi')
def modi():
    id = request.args.get('id')
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    c.execute('SELECT * FROM general WHERE id ='+str(id))   # id 를 설정해주면 설정한 id에 대한 값만 출력. 
    rows = c.fetchall();
    return render_template('board_modi.html', rows = rows)


@app.route('/addrec', methods=['POST', 'GET'])
def addrec() :
    if request.method == 'POST' :
        try:
            war = request.form['war']
            id = request.form['id']
            with sqlite3.connect('./data/wanggun.db') as conn :
                cur = con.cursor()
                cur.execute('update general set war = ' + str(war) + ' where id = '+ str(id))
                conn.commit()
                msg = '정상적으로 입력되었습니다.'
        except : 
            conn.rollback() # 다시 원래대로 돌려라
            msg = '입력과정에서 오류가 발생하였습니다.'
        finally :
            return render_template('board_result.html', msg = msg)
            conn.close()  # 데이터베이스를 닫겠다

@app.route('/addrec', methods=['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            conn = sqlite3.connect('./data/wanggun.db')
            war = request.form['war']
            ids = request.form['id']
            c = conn.cursor()
            c.execute('UPDATE general SET war = '+ str(war) + " WHERE id = "+str(ids))
            conn.commit()
            msg = '정상적으로 입력되었습니다.'
        except:
            conn.rollback()
            msg = '에러가 발생하였습니다.'
        finally:
            conn.close()
            return render_template("board_result.html", msg=msg)


# 실행하시오 !
app.run(host='127.0.0.1', port=5011, debug=False)
'''



'''
from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

# 데이터 베이스
conn = sqlite3.connect('./data/wanggun.db')                 # DB wanggun에 접속
cursor = conn.cursor()
cursor.execute('SELECT * FROM general')                     # general에 있는 모든 데이터 가져오기
print(cursor.fetchall())

@app.route('/')
def run():
    conn = sqlite3.connect('./data/wanggun.db')             # app에서 다시 접속
    c = conn.cursor()
    c.execute('SELECT * FROM general')
    rows = c.fetchall()
    return render_template('board_index.html', rows = rows)

@app.route('/modi')                                         # 이름을 눌렀을 때 전달해주는 구간
def modi():
    id = request.args.get('id')                             # id를 요청하여 넣어준다
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    c.execute('SELECT * FROM general WHERE id ='+str(id))   # WHERE : 선택 
    rows = c.fetchall();
    return render_template('board_modi.html', rows = rows)

@app.route('/addrec', methods=['POST', 'GET'])               # 실질적 수정 부분
def addrec():
    if request.method == 'POST':
        try:                                    
            war = request.form['war']
            id = request.form['id']
            with sqlite3.connect('./data/wanggun.db') as conn:
                cur = conn.cursor()
                cur.execute('UPDATE general SET war='+str(war)+'WHERE id='+str(id)) # war의 컬럼을 수정
                conn.commit()                                                       # 수정 후 항상 커밋
                msg = '정상적으로 입력되었습니다.'
        except:                                             # try 부분에서 error가 뜨면 예외 처리 해준다.
            conn.rollback()                                 # error가 뜨면 rollback한다.
            msg = '입력과정에서 에러가 발생했습니다.'

        finally:
            return render_template('board_result.html', msg = msg)
            conn.close()

app.run(host = '127.0.0.1', port =5000, debug=False)

'''