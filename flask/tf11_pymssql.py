# pip install pymssql
# 접속 : pymssql  방식

# MSSQL 에서 CSV데이터 가져오기

# SQL 실행 시키고
# 포트 확인 : 구성관리자 --> SQL Sever 네트워크 구성 --> 프로토콜
# --> TCP/IP 오른쪽 클릭 --> 속성 --> IP주소 
# --> 맨 아래(IPAll) TCP 동적포트 숫자 지우고(적어두기)
# --> TCP 포트에 1433 넣고 --> 적용 --> 확인
# --> 작업관리자 --> 서비스 --> MSSQL$ ~ 라고 써 있는걸 오른쪽 클릭 --> 재시작
# 또는 MyFw40Service 오른쪽 클릭 --> 재시작

import pymssql as ms

# print('잘 접속됐지')

conn = ms.connect(server = '127.0.0.1', user = 'bit2', 
                  password = '1234', database = 'bitdb')
                  #port = 49683)

cursor = conn.cursor()

cursor.execute("SELECT * FROM iris2;")  # 실행
# cursor.execute("SELECT * FROM wine;")  # 실행
# cursor.execute("SELECT * FROM sonar;")  # 실행

row = cursor.fetchone()

while row :
    print("첫컬럼 : %s, 둘컬럼 : %s" %(row[0], row[1]))
    # print("첫컬럼 : %s, 둘컬럼 : %s, 셋컬럼 : %s, 넷컬럼 : %s" %(row[0], row[1], row[2], row[3]))
    row = cursor.fetchone()

conn.close()





