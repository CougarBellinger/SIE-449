import numpy as np
import numpy.linalg as la

#Initialize matrix A and vector b with fixed random numbers
np.random.seed(123)
n = 10
R = np.random.randn(n,n)
A = R.T @ R
b = np.random.randn(n, 1)

#Initialize variables
x = np.zeros((n, 1))
e = 1e-3
L = 2 * la.norm(A,2)
alpha = 1 / L

#returns projected value
def projection(x, c = -10):
    sum = np.sum(x)
    
    if sum >= c:
        return x
    else:
        diff = c - x
        adj = diff / len(x)
        projected = x + adj
        return projected

#Begin algorithm
for _ in range(100000): # range used to prevent infinite loop
    grad = 2 * (A@x + b)
    x_temp = x - alpha * (grad)
    x = projection(x_temp)
    
    if(la.norm(x - x_temp, 2) <= e):
        fun_val = x.T @ A @ x + 2*(b.T @ x)
        min_val = x
        break

print("Value found:")
print(min_val)
print("Function value:")
print(fun_val)