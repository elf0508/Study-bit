# 시험점수를 입력받아 90~100점은 A
# 80~89점은 B,  70~79점은 C  , 60~69점은 D,  나머지는 점수는 F 를 출력

score = int(input())

if score >= 90:     # 90점 이상
    print("A")
elif score >= 80:   # 80~89
    print("B")
elif score >= 70:   # 70~79
    print("C")
elif score >= 60:   # 60~69
    print("D")
else:               # 그 외의 경우, 59점 이하
    print("F")