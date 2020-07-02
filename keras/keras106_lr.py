# lr = learning_rate

weight = 0.5  # 시작점
input = 0.5
goal_prediction = 0.8   # 예측값(목표점)

lr = 0.001  # Error : 1.0799505792475652e-27  Prediction : 0.7999999999999672 
# lr = 0.1      # Error : 0.0024999999999999935   Prediction : 0.7500000000000001
# lr = 1      # Error : 0.20249999999999996     Prediction : 1.25
# lr = 0.0001   # Error : 0.24502500000000604     Prediction : 0.30499999999999394
# lr = 100        # Error : 0.30250000000000005     Prediction : 0.25


for iteration in range(1101):
    prediction = input * weight
    # 0.8 = input * weight
    error = (prediction - goal_prediction) **2  # **2 = 제곱
    # error = loss

    print("Error : " + str(error) + "\tPrediction : " + str(prediction))
                    # error = loss
    up_prediction = input * (weight + lr)
    up_error = (goal_prediction - up_prediction) **2

    down_prediction = input * (weight - lr)
    down_error = (goal_prediction - down_prediction) **2

    if(down_error < up_error):
        weight = weight - lr

    if(down_error > up_error):
        weight = weight + lr
    
'''
weight = 0.5           # 초기 가중치
input = 0.5            # x
goal_prediction = 0.8  # y

lr = 0.001

for iteration in range(1101):                           # 0.5를 넣어서 0.8을 찾아가는 과정
    prediction = input * weight                         # y = w * x 
    error = (prediction - goal_prediction)**2           # loss

    print('Error : ' + str(error)+'\tPrediction : '+str(prediction))

    up_prediction = input *(weight + lr)                # weight = gradient : -경사 올림
    up_error = (goal_prediction - up_prediction)**2     # loss

    down_predicrion = input*(weight - lr)               # weight = gradient : +경사 내림
    down_error = (goal_prediction - down_predicrion)**2 # loss

    if(down_error < up_error):                          
        weight = weight - lr                            

    if(down_error > up_error):                          
        weight = weight + lr                            

'''