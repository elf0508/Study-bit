# for문 : 반복문

a = {'name' : 'yun', 'phone' : '010', 'birth' : '0511'}
 # a.keys()값을 하나씩 i에 넣어라
for i in a.keys():
    print(i)  # 순서대로 출력된다 / name  phone  birth

a = [1,2,3,4,5,6,7,8,9,10]
for i in a:
    i = i*i
'''
4
9
16
25
36
49
64
81
100 로 나온다
'''

for i in a:    
    print(i)
'''
2
3
4
5
6
7
8
9
10  로 나온다
'''    
# print('melong')

## while문
'''
while 조건문 : '참'일 동안 계속 된다
    수행할 문장
'''

### if문
 
if 1 :
     print('True')  # True
else:
    print('False')

if 3 :
     print('True')  # True
else:
    print('False')

if 0 :
     print('True')  # False
else:
    print('False')

if -1 :
     print('True')  # True
else:
    print('False')

'''
비교 연산자

< , > , == , !=, >= , <=
        
'''
a = 1
if a == 1 :
    print('출력잘되')  #  출력잘되

money = 10000
if money >= 30000:
    print('한우먹자')
else:
    print('라면먹자')  #  라면먹자

### 조건 연산자
# and, or, not
money = 20000
card = 1
if money >= 30000 or card == 1:
    print('한우먹자') # 한우먹자
else:
    print('라면먹자')

jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:
    if i >=60:
        print("경] 합격 [축")
        number = number + 1
    
print("합격인원 : " , number, "명")
'''
경] 합격 [축
경] 합격 [축
경] 합격 [축
합격인원 :  3 명  으로 나온다
'''
print("======= break ===================")
# break, continue

jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:
    if i < 30:
        break   #  제일 가까운 for문을 중지 시킨다
    if i >=60:
        print("경] 합격 [축")
        number = number + 1
    
print("합격인원 : " , number, "명")
'''
경] 합격 [축
합격인원 :  1 명 으로 나온다
'''
print("======== continue ===============")
# break, continue

jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:
    if i < 60:
        continue   
    if i >=60:
        print("경] 합격 [축")
        number = number + 1
    
print("합격인원 : " , number, "명")
'''
경] 합격 [축
경] 합격 [축
경] 합격 [축
합격인원 :  3 명 으로 나온다
'''
