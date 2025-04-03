import numpy as np
x = 4
for k in range(1,21):
    step = 0.1
    grad = 2 * x
    x = x - step * grad

print(x)

