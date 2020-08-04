# PCA

import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

dataset = load_diabetes()  #  컬럼 총 10개

X = dataset.data
Y = dataset.target

print(X.shape)   # (442, 10)
print(Y.shape)   # (442,   )

# pca = PCA(n_components = 5)  # 중요한거 순으로 5개로 압축
# # pca = PCA(n_components = 10)  
# x2 = pca.fit_transform((X))
# pca_evr = pca.explained_variance_ratio_

# print(pca_evr)   # [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856]  
# # 압축한 컬럼들의 중요도

# print(sum(pca_evr))   # 0.8340156689459766

pca = PCA()
pca.fit(X)

cumsum = np.cumsum(pca.explained_variance_ratio_)
# cumsum : 누적 계산

print(cumsum)

#  [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#   0.94794364 0.99131196 0.99914395 1.  ]


aaa = np.argmax(cumsum >= 0.94)+1   
# +1 하는 이유 : 94% 압축률을 가진PCA를 사용 하고싶을때, 7개의 특성으로 압축을 하고싶은때
# 인덱스가 0부터 시작이기 때문에 +1을 해준다.
# argmax : 최댓값

print(cumsum >= 0.94)  # [False False False False False False  True  True  True  True]

print(aaa)             # 7  

