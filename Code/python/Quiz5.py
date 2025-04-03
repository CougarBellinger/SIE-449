### Complete the code and upload the code and the plot in D2L ###

### Nesterov's Accelerated Gradient Method ###
### Solving min ||Ax-b||^2 when the size of A is mxn, x is nx1 and b is mx1 ###  

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)  # fixing seed to generate same random number every time

m = 4
n = 3
x = np.zeros((n, 1))  # initial point
xold = x.copy()  # initial point
A = np.random.rand(m, n)  # defining matrix A
b = np.random.rand(m, 1)  # defining vector b
maxiter = 200  # total number of iterations 
vec = []

for k in range(1, maxiter + 1):  
    # 2AᵀAx - 2Aᵀb
    L = 2*(np.linalg.norm(A))**2		# Lipschitz constant of the gradient
    alpha = 1 / L 		# stepsize
    y = x + ((k - 2) / (k + 1)) * (x - xold) 			# acceleration step
    # grad = 2*A'*(A*y-b)
    grad = 2*A.T@(A@y-b)	# computing the gradient at y
    xold = x.copy()  	# save previous iterate
    x = y - alpha*grad			# gradient step
    vec.append(np.linalg.norm(grad))  # save norm(grad)

plt.semilogy(vec)
plt.show()
