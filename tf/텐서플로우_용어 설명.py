'''

텐서플로우 tf17_mnist 


[ 플레이스홀더 ]

딥러닝에서 데이터에 대한 학습이 이루어질 때 학습할 데이터들을 입력해줘야 한다. 

예를 들어, MNIST 데이터를 학습한다고 할때 입력값으로 MNIST 이미지 데이터를 입력값으로 넣어 줘야 했다. 

텐서플로에서는 입력값을 넣어주기 위해 플레이스홀더(placeholder)라는 것이 있다. 

플레이스홀더는 데이터를 입력받는 비어있는 변수라고 생각할 수 있다. 

먼저 그래프를 구성하고, 그 그래프가 실행되는 시점에 입력 데이터를 넣어주는 데 사용한다.

플레이스홀더는 shape 인수를 유동적으로 지정할 수 있다. 

예를 들어, None으로 지정되면 이 플레이스홀더는 모든 크기의 데이터를 받을 수 있다. 

주로 배치단위(batch size)의 샘플 데이터 개수에 해당 되는 부분(데이터의 행)은 None을 사용하고, 데이터 Feature의 길이(데이터의 열)는 고정된 값을 사용한다. 

예) ph = tf.placeholder(tf.float32, shape=(None, 10))

플레이스홀더를 정의하면 반드시 그래프 실행 단계에서 입력값을 넣어줘야 하며, 그렇지 않을 경우 에러가 나타난다. 

입력 데이터는 딕셔너리(dictionary)형태로 session.run()메소드를 통해 전달된다. 

딕셔너리의 키(key)는 플레이스홀더 변수 이름에 해당하며 값(value)은 list 또는 NumPy 배열이다.

예) sess.run(s, feed_dict={ph: data})


[ 레이어 ]

tf.Variable()

1] 언제나 새로운 객체를 만들어 낸다.
이미 같은 이름의 객체가 있다면, _1, _2 등을 붙여서 유니크(uniqu)하게 만든다.

2] 초기값을 지정해야 한다.

-----------------------------------------------

tf.get_variable()

1] 이미 존재하는 객체(name filed)를 매개 변수로 받을수도 있다.
(변수는 같은 객체를 가르키게 된다. --> 재활용 가능)
그런데, 해당 객체가 없다면 새로운 객체로 만들어 낸다.

tf.get_variable()를 사용하는 것이 좀 더 범용적이라고 할 수 있다.

-------------------------------------------------------------

tf.get_variable() 함수의 인자

def get_variable(name,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=True,
                 collections=None,
                 caching_device=None,
                 partitioner=None,
                 validate_shape=True,
                 custom_getter=None):

-------------------------------------------

1. name

텐서의 name filed에 들어가는 값이다. 이 값이 동일한 텐서가 있으면 해당 텐서를 리턴해주고, 없으면 새로 생성한다.


2. shape, dtype

리턴할 텐서의 shape과 데이터 타입


3. initializer

텐서 초기화할 때 사용될 op

사용가능한 initializer로는 tf.zeros_initializer, tf.ones_initializer(), tf.truncated_normal_initializer(), tf.contrib.layers.xavier_initializer(), tf.contrib.layers.variance_scaling_initializer() 등이 있다.


4. regularizer

이 텐서에 적용될 정규화 방법론이 무엇이냐를 정해주는 것이다.

가장 많이 사용되는 l2정규화를 예를 들면, 다음과 같은 op를 인자로 넘겨주면 된다.

regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

여기서 weight_decay는 정규화에 곱해지는 상수값이다.

l2의 경우, 텐서의 모든 인자값의 제곱의 합에 weight_decay를 곱한 값이 정규화에 사용된다.

이 인자가 선언 되면, 정규화값을 위해 variable이 추가로 선언이 되고, 이 variable은 자동으로 collection의 일종인 GraphKeys.REGULARIZATION_LOSSES에 소속이 된다


5. trainable

이 variable을 학습을 통해 값을 변화시킬 것인지에 대한 인자이다. 

True가 되면 자동을 collection의 일종인 GraphKeys.TRAINABLE_VARIABLES에 소속이 된다. 

이 값이 False이면 학습 중에 값이 변하지 않으므로, tf.constant()처럼 사용하게 되는 것이다.


6. collections

collection은 variable의 소속이라고 생각하면 된다. 그리고 그 소속은 여러 곳일 수 있다.

그래서 collections로 넘어오는 인자는 생성될 variable이 소속될 collection에 대한 리스트이다.

즉, collections=[collection1, collection2]의 값을 넣어주면 이 variable은 collection1에도 소속이 되고, collection2에도 소속이 된다.

그럼 이것을 왜 정해주냐인데.. 주 목적은 해당 variable을 코드의 다른 위치에서 불러오기 위해서이다.

variable_scope과 get_variable()함수의 조합은 name filed값을 기억해야 하나의 variable을 불러올 수 있지만, 특정 목적을 위한 variable의 집합을 불러올 때는 collection과 tf.get_collection()의 조합으로 가능하다. 

tf.get_collection(key)가 실행되면, key의 collection에 속하는 variable들의 리스트가 리턴된다.

많이 사용되는 key는 tf.GraphKeys에 선언이 되어 있다. (앞의 예에서 GraphKeys.REGULARIZATION_LOSSES, GraphKeys.TRAINABLE_VARIABLES 등)

본인이 직접 string으로 넣어주어도 된다.

-------------------------------------------------

[ tf.get_collection() 함수와 관련된 다양한 사용법 ]


1. regularization

기존 loss값에 정규화값을 추가하고 싶을 경우, 정규화가 적용될 variable을 get_variable(..., regularizer = 방법)으로 

GraphKeys.REGULARIZATION_LOSSES에 속하게 만든다.

그리고 tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)을 수행하면, 정규화값들의 리스트가 리턴된다.

그래서, tf.reduce_sum()으로 하나의 정규화값을 만든 후, 기존 loss값에 대해주면 되는 것이다.

loss = tf.add(loss, tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))


2. tf.add_to_collection(key, variable)

get_variable()에서 variable을 collecton에 소속시키지 않았는데 새로 소속을 시키거나, 추가로 소속시킬 때 이 함수를 통해서 소속시킬 수 있다.


3. histogram_summary

학습이 될 variable에 대해서 histogram을 summary하는 것이 다음 코드로 가능하다.

for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

여기서 tf.trainable_variables()는 내부적으로 get_collection(GraphKeys.TRAINABLE_VARIABLES)가 호출된다.


4. tf.summary.merge_all()

summary를 생성할 때 보통 summary_op = tf.summary.merge_all()를 선언한 뒤, add_summary(sess.run(summary_op))하게 된다.

이때, summary에 관한 op들은 학습과 무관하므로, 별도로 실행줘야 하는데, 

이때 사용되는 tf.summary.merge_all()함수는 내부적으로 get_collection(GraphKeys.SUMMARIES)가 호출된다. 

즉, tf.summary.~로 생성되는 variable들은 default로 collection이 GraphKeys.SUMMARIES가 된다는 것을 알 수 있다. 

하나의 예로, tf.summary.scalar()에 대한 설명에 다음이 있다.


def scalar(name, tensor, collections=None):

  """Outputs a `Summary` protocol buffer containing a single scalar value.

  The generated Summary has a Tensor.proto containing the input Tensor.

  Args:

    name: A name for the generated node. Will also serve as the series name in
      TensorBoard.
    tensor: A real numeric Tensor containing a single value.
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.



5. tf.train.Saver()

학습된 모델을 저장하는 코드는 주로 다음과 같다.

saver = tf.train.Saver(tf.global_variables())

saver = tf.train.Saver()

tf.train.Saver()의 인자는 저장할 variable들의 리스트가 들어간다. 아무것도 넣어주지 않으면 저장가능한 모든 variable들이 default로 인자로 넘어간다.

tf.global_variables()의 내부적으로는 get_collection(GraphKeys.GLOBAL_VARIABLES)이 호출되며, 

GrapheKyes.GLOBAL_VARIABLES는 variable이 선언될 때 default로 소속이 되는 collection이다.


6. tf.global_variables_initializer()


자. 그럼 이 함수는 내부적으로 어떻게 실행이 될까? 

get_collection(GraphKeys.GLOBAL_VARIABLES)로 모든 variable의 리스트를 받아와 각 variable의 initializer에 대한 op를 하나로 묶어서 리턴해준다. 

그래서, sess.run(tf.global_variables_initializer()) 하는 것이 variable를 초기화 주는 코드가 되는 것이다.


7. batch_norm()

batch_norm()에서는 test할 때 사용할 moving_mean과 moving_var이 train할 때 계산을 해줘야 한다.

즉, train과정에서 moving_mean과 moving_var이 직접적으로 호출이 되지 않기 때문에 train과 별도로 이 moving_mean과 moving_var에 대한 op를 실행줘야 한다.

달리 표현하면, loss계산 시, moving_mean과 moving_var이 계산되지 않기 때문에 train과는 별도로 sess.run()에서 호출해줘야 하는 것이다.

이를 위해, batch_norm()의 인자들 중, updates_collections=ops.GraphKeys.UPDATE_OPS가 있다. 

moving_mean과 moving_var에 대한 op의 collection을 GraphKeys.UPDATE_OPS로 지정하여, 추후 tf.get_collection(tf.GraphKeys.UPDATE_OPS)로 이와 관련된 op들을 넘겨받아, sess.run()에서 실행하면 되는 것이다.


===================================================================================================

[ 텐서플로우 연산 ]

tf.random_normal(shape, mean, stddev) : 형태((shape), 평균(mean), 표준편차(stddev) 를 넣어서 정규분포를 따르는 난수들을 생성

tf.zeros(shape) : (    )에 지정된 형태(shape)의 텐서를 만들고, 모든 원소의 값을 '0' 으로 초가화 한다.

========================================================================

[ 행렬곱 ]

텐서플로에서 행렬곱은 tf.matmul() 함수를 이용하여 연산을 수행한다. 

예를 들어, 두 텐서 객체 A와 B의 행렬곱은 tf.matmul(A, B)로 계산할 수 있다.

=========================================================

[ dropout ]

over-fitting을 줄이기 위한 regularization 기술이다.

네트워크에서 일시적으로 유닛(인공 뉴런, artificial neurons)을 배제하고, 그 배제된 유닛의 연결을 모두 끊는다.

텐서플로우에선 드롭아웃을 구현할 때 tf.nn.dropout() 함수를 사용한다.

예)

keep_prob = tf.placeholder(tf.float32) # probability to keep units
 
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)
 
logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
 
    for epoch_i in range(epochs):
        for batch_i in range(batches):
            ....
 
            sess.run(optimizer, feed_dict={
                features: batch_features,
                labels: batch_labels,
                keep_prob: 0.5})
 
    validation_accuracy = sess.run(accuracy, feed_dict={
        features: test_features,
        labels: test_labels,
        keep_prob: 0.5})


 - tf.nn.dropout()함수는 두 인자를 받는데,

 - hidden_layer : dropout을 적용할 텐서

 - keep_prob : 주어진 유닛을 유지할 확률 즉, drop하지 않을 확률

 - keep_prob을 사용하면 drop할 유닛의 수를 조절할 수 있다.

 - drop된 유닛을 보상하기위해 tf.nn.dropout() 함수는 drop되지 않은 모든 유닛에 1 / keep_prob을 곱한다.

 - 학습을 진행할 땐 무난한 keep_prob 값은 0.5이다.

 - 테스트를 진행할 땐 keep_prob 값을 1.0으로 두어 모든 유닛을 유지하고 모델의 성능을 극대화한다.


------------------------------------------------------------------------------

예)

- ReLU 그리고 dropout 레이어를 적용해보도록 하자.

 - ReLU 레이어와 dropout 레이어를 사용하여 모델을 만들자.

 - 이 때, dropout 레이어의 keep_prob placeholder는 0.5로 셋팅한다.

 - 그리고 모델의 logits을 출력한다.

 - 코드가 실행될 때 마다 output이 달라지는데 그 이유는 dropout한 유닛이 랜덤하게 변하기 때문이다.


import tensorflow as tf
 
hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]
 
# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]
 
# Input
features = tf.Variable([[0.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4], [11.0, 12.0, 13.0, 14.0]])
 
# TODO: Create Model with Dropout
keep_prob = tf.placeholder(tf.float32)
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)
 
logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])
 
# TODO: Print logits from a session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(logits, feed_dict={keep_prob: 0.5}))

'''






