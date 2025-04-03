# https://d2l.arizona.edu/d2l/le/content/1546967/viewContent/18374082/View

import numpy as np
import numpy.linalg as la

m = 3
n = 2

A = np.array([[1,4], [4.5,4], [3,5]])
b = np.array([[2.5], [5], [4]])
cafe = np.array([3,3])

x = np.zeros((n, 1))
e = 1e-5

grad = 2*A.T@(A@x - b) # Use @ when multiplying vector and matrices
alpha = 1 / (2 * la.norm(A)**2)

# Gradient Method
while (la.norm(grad)) > e:
    # Limpschitz constant of the gradient
    L = 2 * la.norm(A,2)**2
    # Stepsize
    alpha = 1 / L 
    # Gradient step
    x = x - alpha * grad
    grad = 2 * A.T@(A@x - b)

print("Vector x: ", x)
print('Norm of the gradient = {0:.5f}'.format(la.norm(grad)))
print('Function value = {0:.5f}'.format(la.norm(A@x-b)**2))

print("Prediction of Sarah's rating:", ((cafe.T)@x))