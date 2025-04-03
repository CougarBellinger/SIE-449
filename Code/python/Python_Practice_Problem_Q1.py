## Solving min x'Ax+2b'x ##

import numpy as np
import numpy.linalg as la

def gradient_method_quadratic(A,b,x0,epsilon):
    x = x0  # initialization
    grad = 2 * (A @ x + b) # compute the initial gradient
    L = 2 * la.norm(A,2)  # Lipschitz constant of the gradient 
    alpha = 1 / L  # stepsize
    
    # Iterate until the gradient norm is smaller than the given threshold
    while la.norm(grad) > epsilon:
        x = x - alpha * grad # gradient step
        grad = 2 * (A @ x + b)  # compute the gradient
        fun_val = x @ A @ x + 2 * b @ x # compute the function value
    return x, fun_val  # Return the optimal solution and function value

# Define problem parameters
A = np.array([[1, 0], [0, 2]])  # defining matrix A
b = np.array([0, 0])  # defining vector b
x0 = np.array([2, 1])  # initial point 
epsilon = 1e-5  # tolerance

# Run the gradient descent method and get the result
x, fun_val = gradient_method_quadratic(A, b, x0, epsilon)
print("x is", x,"function value is", fun_val)