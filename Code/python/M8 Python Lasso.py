###Solving min 1/2||Ax-b||^2+lambda*||x||_1 whene the size of A is mxn, x is nx1 and b is mx1###

import numpy as np

np.random.seed(123)  # fixing seed to generate same random number every time

m = 10  # set the size of the matrix
n = 10  # set the size of the matrix
x = np.zeros((n, 1))  # initial point
A = np.random.rand(m, n)  # defining matrix A
b = np.random.rand(m, 1)  # defining vector b
maxiter = 1000 # maximum number of iterations
lmbda = 0.03  # tuning parameter
# LOOKUP: How to choose lambda. How to choose norm type
for k in range(1, maxiter + 1):  # stopping criteria
    L = np.linalg.norm(A)**2  # Lipschitz constant of the gradient CHANGE THIS FOR Q5 on HW4
    alpha = 1 / L  # stepsize
    grad = A.T @ (A @ x - b)  # computing the gradient CHANGE THIS FOR Q5 on HW 4
    x = x - alpha * grad  # gradient step
    x = np.maximum(np.abs(x) - alpha * lmbda * np.ones((n, 1)), 0) * np.sign(x)  # proximal step

print(f'Value of the vector x:\n{x.flatten()}\n number of zeros = {np.sum(x == 0)}')  # printing the last iterate point and number of zeros in x
