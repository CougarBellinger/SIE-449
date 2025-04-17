import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load data
X = loadmat('train_rating.mat')['X']
y = loadmat('train_labels.mat')['y']
groupLabels = loadmat('train_grouplabels.mat')['groupLabels'].flatten()

# Data preparation
n, p = X.shape
X = np.hstack((np.ones((n, 1)), X))  # Add intercept
p = X.shape[1] - 1  # updated p (excluding intercept)

# Parameters
L = (np.linalg.norm(X,2) ** 2 / 4) / n
alpha = 2 / L
lmbda = 1 / n
maxiter = 100
beta0 = np.zeros((p + 1, 1))

# Logistic loss function
def func(x):
    return 1 / n * (np.sum(np.log(1 + np.exp(X @ x))) - y.T @ (X @ x))

# Gradient of logistic loss
def grad(x):
    exp_term = np.exp(X @ x)
    return 1 / n * (-X.T @ y + X.T @ (exp_term / (1 + exp_term)))

# Group Lasso proximal operator
def prox_group(beta, group_labels, alpha, lmbda):
    prox = []
    J = np.max(group_labels)
    for j in range(1, J + 1):
        ind_j = (group_labels == j)
        beta_j = beta[ind_j]
        norm_j = np.linalg.norm(beta_j)
        p_j = np.sum(ind_j)
        if norm_j != 0:
            scale = max(0, 1 - alpha * lmbda * np.sqrt(p_j) / norm_j)
        else:
            scale = 0
        prox.append(scale * beta_j)
    return np.concatenate(prox).reshape(-1, 1)

# Group regularization term
def reg(beta, group_labels):
    out = 0
    J = np.max(group_labels)
    for j in range(1, J + 1):
        ind_j = (group_labels == j)
        p_j = np.sum(ind_j)
        out += np.sqrt(p_j) * np.linalg.norm(beta[ind_j])
    return out

fopt = 213.741 / n  # Optimal value (given)

# Proximal Gradient Descent
beta = beta0.copy()
vec_pg = []
for k in range(1, maxiter + 1):
    beta = beta - alpha * grad(beta)
    beta[1:] = prox_group(beta[1:], groupLabels, alpha, lmbda)
    vec_pg.append(func(beta) + lmbda * reg(beta[1:], groupLabels))
    
plt.semilogy(np.array(vec_pg).flatten() - fopt, label='Proximal Gradient')

# Accelerated Proximal Gradient (Nesterov)
beta = beta0.copy()
betaold = beta.copy()
vec_acc = []
for k in range(1, maxiter + 1):
    z = beta + (k - 2) / (k + 1) * (beta - betaold)
    betaold = beta.copy()
    beta = z - alpha * grad(z)
    beta[1:] = prox_group(beta[1:], groupLabels, alpha, lmbda)
    vec_acc.append(func(beta) + lmbda * reg(beta[1:], groupLabels))

plt.semilogy(np.array(vec_acc).flatten() - fopt, label='Nesterov Acceleration')

# Plot
plt.xlabel('Iteration')
plt.ylabel(r'$f(x_k) - f^*$')
plt.title('Comparison of Proximal and Accelerated Proximal Gradient Methods')
plt.grid(True)
plt.legend()
plt.show()
