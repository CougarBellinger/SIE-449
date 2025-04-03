# https://d2l.arizona.edu/d2l/le/content/1546967/viewContent/18374082/View

import numpy as np
import numpy.linalg as la

np.random.seed(123)
m = 4
n = 3

x = np.zeros((n, 1))
e = 1e-5

A = np.random.rand(n,m) # Flip generation to match matlab
A = A.T

b = np.random.rand(4,1)

grad = 2*A.T@(A@x - b) # Use @ when multiplying vector and matrices
alpha = 1 / (2 * la.norm(A)**2)

# Gradient Method
while (la.norm(grad)) > e:
    # Limpshitz constant of the gradient
    L = 2 * la.norm(A,2)**2
    # Stepsize
    alpha = 1 / L 
    # Gradient step
    x = x - alpha * grad
    grad = 2 * A.T@(A@x - b)

print("Vecotr x: ", x)
print('norm of the gradient = {0:.5f}'.format(la.norm(grad)))
print('function value = {0:.5f}'.format(la.norm(A@x-b)**2))