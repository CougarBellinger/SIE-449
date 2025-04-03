# !pip install qpsolvers
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

# Set seed
np.random.seed(123)

# Generating data points
d = 50
n = 2
slope = np.random.rand(n)
intercept = 0
X = np.random.randn(d, n)
y = np.ones(d)
y[X.dot(slope) + intercept < 0] = -1

# Initialize alpha
alpha = np.zeros(d)
tol = 1e-3
maxiter = int(1e4)
step_size = 1 / np.linalg.norm(X)**2

obj_values = np.zeros(maxiter)

# Main loop
for iter in range(maxiter):
    # Compute gradient
    Q = (np.diag(y) @ X) @ X.T @ np.diag(y)
    grad = Q * alpha - np.ones(d)

    # Store last iter
    alpha_old = alpha

    # Update alpha using gradient step
    alpha = alpha - step_size * (grad)

    # Project alpha onto the feasible set {alpha>=0 , sum(y.*alpha)==0}
    P = np.eye(d)
    q = -alpha
    G = None  
    h = None
    A = y.T
    b = np.array([0.0])
    lb =  np.zeros(d)
    ub = None
    alpha = solve_qp(P, q, G, h, A, b, lb, ub, solver="cvxopt")

    # Compute objective value
    obj_values[iter] = 0.5 * alpha.T @ Q @ alpha - np.sum(alpha)

    # Check convergence
    if np.linalg.norm(alpha - alpha_old) < tol:
        print('Accuracy achieved!')
        break

# !!!!!!WRITE SLOPE AND INTERCEPT HERE!!!!!!
w = X.T @ (y * alpha)
ind = np.argmax(alpha)
beta = y[ind] - X[ind, :] @ w

# Display final objective value
print('Final Objective Value:', obj_values[iter])

# Display the dataset
plt.figure()
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', label='+1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='x', label='-1')
plt.title('Linearly Separable Dataset for Hard-Margin SVM')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

vec = np.linspace(-2, 2, 2)
plt.plot(vec, -(vec * slope[0] + intercept) / slope[1], '-', label='true classifier')
plt.plot(vec, -(vec * w[0] + beta) / w[1], '-', label='proj grad')
plt.grid(True)
plt.legend(loc='best')
plt.show()



