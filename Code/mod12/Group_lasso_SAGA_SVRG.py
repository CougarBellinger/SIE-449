import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat

# Fixing the random seed for reproducibility
np.random.seed(123)

# Load data
X = loadmat('train_rating.mat')['X']
y = loadmat('train_labels.mat')['y']
groupLabels = loadmat('train_grouplabels.mat')['groupLabels'].flatten()

# Data preparation
n, p = X.shape
X = np.hstack((np.ones((n, 1)), X))  # Add intercept
p = X.shape[1] - 1  # updated p (excluding intercept)

# Parameters
L = (np.max(np.linalg.norm(X, axis=0)) ** 2) / 4
alpha = 1 / (3*L)
lmbda = 1 / n
maxiter = 100*n
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

# SAGA
beta = beta0.copy()
vec_saga = [(func(beta) + lmbda * reg(beta[1:], groupLabels)).item()]
gradtable = np.zeros((p + 1, n))  # Each column corresponds to gradient at sample i


start_time = time.time()
for i in range(n):
    xi = X[i, :].reshape(1, -1)  # row vector
    yi = y[i, 0]
    grad_i = (-xi.T * yi + xi.T * (np.exp(xi @ beta) / (1 + np.exp(xi @ beta)))).reshape(-1)
    gradtable[:, i] = grad_i


for k in range(1, maxiter + 1):
    i = np.random.randint(n)  # randomly sample i in [0, n-1]
    xi = X[i, :].reshape(1, -1)
    yi = y[i, 0]

    exp_term = np.exp(xi @ beta)
    gradi = (-xi.T * yi + xi.T * (exp_term / (1 + exp_term))).reshape(-1)
    grad = gradi - gradtable[:, i] + np.mean(gradtable, axis=1)
    gradtable[:, i] = gradi

    beta = beta - alpha * grad.reshape(-1, 1)
    beta[1:] = prox_group(beta[1:], groupLabels, alpha, lmbda)

    if k % (2*n) == 0:
        val = (func(beta) + lmbda * reg(beta[1:], groupLabels)).item()
        vec_saga.append(val)

timesaga = time.time() - start_time
print(f"SAGA Elapsed time: {timesaga:.4f} seconds")

plt.semilogy(np.array(vec_saga) - fopt, label='SAGA')

# SVRG 
beta = beta0.copy()
vec_svrg = [(func(beta) + lmbda * reg(beta[1:], groupLabels)).item()]
tildebeta = beta.copy()

start_time = time.time()  # Start timing

for s in range(50):  # Outer loop
    exp_term = np.exp(X @ tildebeta)
    tildeg = 1 / n * (-X.T @ y + X.T @ (exp_term / (1 + exp_term)))

    inner = 2 * n  # Number of inner iterations
    betasum = np.zeros_like(beta)

    for m in range(inner):  # Inner loop
        i = np.random.randint(n)
        xi = X[i, :].reshape(1, -1)
        yi = y[i, 0]

        exp_b = np.exp(xi @ beta)
        gradi = (-xi.T * yi + xi.T * (exp_b / (1 + exp_b))).reshape(-1)

        exp_tb = np.exp(xi @ tildebeta)
        gradtilde = (-xi.T * yi + xi.T * (exp_tb / (1 + exp_tb))).reshape(-1)
        grad = gradi - gradtilde + tildeg.flatten()

        beta = beta - alpha * grad.reshape(-1, 1)
        beta[1:] = prox_group(beta[1:], groupLabels, alpha, lmbda)
        betasum += beta

    tildebeta = betasum / inner  # Average of iterates

    val = (func(tildebeta) + lmbda * reg(tildebeta[1:], groupLabels)).item()
    vec_svrg.append(val)
    
timesvrg = time.time() - start_time
print(f"SVRG Elapsed time: {timesvrg:.4f} seconds")

plt.semilogy(np.array(vec_svrg) - fopt, label='SVRG')


# Plot
plt.xlabel('Iteration')
plt.ylabel(r'$f(x_k) - f^*$')
plt.title('Comparison of SAGA and SVRG')
plt.grid(True)
plt.legend()
plt.show()
