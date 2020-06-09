import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9,10])
print(x)
print(x.shape)

x = np.transpose([1,2,3,4,5,6,7,8,9,10])

print("=== transpose ===")

# np = np.transpose(x)
print(x)
print(x.shape)


print(x)
print(x.shape)

print("=== reshape 1 ===")
x = x.reshape(5, 2)
print(x)
print(x.shape)

print("=== reshape 2 ===")
x = x.reshape(2, 5)
print(x)
print(x.shape)
