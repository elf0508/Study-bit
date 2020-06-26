
# 함수 : 인수를 받아 처리한 결과를 반환값으로 돌려준다.

# def 함수명(인수):

# 함수를 호출할 때는 함수명(인수)을 사용한다.
# 함수는 정의를 한 후에만 사용 가능하다.

def sing():
    print("노래합니다!!)

sing()

# 문자열의 모맷 지정

# 문자열형 메서드 : format()

# % 연산자 

# %d : 정수로 표시

# %f : 소수로 표시

# %.2f : 소수점 이하 두 자리까지 표시

# %s : 문자열로 표시

# 예) 이진 검색 알고리즘을 이용하여 검색하는 프로그램

# 함수 brinary_search의 내용

def brinary_search(number, target_number):

    # 최소값을 임의로 설정
    low = 0

    # 범위내의최대값
    high =len(number)

    # 목적지를 찾을 때까지 무한루프
    while low <= high:

        # 중앙값을 구한다.(index)
        middle = (low + high) // 2

        # numbers(검색 대상)의 중앙값과 target_number(찾는 값)가 동일한 경우
        if number[middle] == target_number:
            print("{1}은 {0}번째에 있습니다.".format(middle, target_number))

            break
        
        # numbers의 중앙값이 target_number보다 작을 경우
        elif number[middle] < target_number:
            low = middle + 1

        # numbers의 중앙값이 target_number보다 클 경우
        else:
            high = middle - 1

# 검색 대상 데이터

number = [1,2,3,4,5,6,7,8,9,10,11,12,13]

# 찾을 값
target_number = 11

# 바이너리 실행
brinary_search(number, target_number)


