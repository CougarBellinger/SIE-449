### Solving min ||Ax-b||^2 whene the size of A is nxm, x is mx1 and b is nx1 via SAG/SAGA/SG ###

import numpy as np
import matplotlib.pyplot as plt

# Fixing the random seed for reproducibility
np.random.seed(123)

n = 1000  # Set the size of the matrix
m = 50  # Set the size of the matrix
x = np.zeros((m, 1))  # Initial point
A = np.random.randn(n, m)  # Defining matrix A
A = A / np.linalg.norm(A)
b = A@(np.random.randn(m, 1)) + np.random.randn(n, 1)  # Defining vector b

maxiter = 10000
L = 2 * n * np.linalg.norm(A,2) ** 2

### ---------- SAG ---------- ###
x = np.zeros((m, 1))
gradtable = []

for i in range(n):
    Ai = A[i, :].reshape(1, -1)  # row vector
    bi = b[i]
    grad_i = 2 * n * Ai.T @ (Ai @ x - bi)
    gradtable.append(grad_i)

gradtable = np.hstack(gradtable)  # (m, n)

vec = []
for k in range(1, maxiter + 1):
    alpha = 10 / (16 * L)
    i = np.random.randint(n)
    Ai = A[i, :].reshape(1, -1)
    bi = b[i]
    gradi = 2 * n * Ai.T @ (Ai @ x - bi)
    grad = (1/n) * gradi - (1/n) * gradtable[:, [i]] + (1/n) * np.sum(gradtable, axis=1, keepdims=True)
    gradtable[:, [i]] = gradi
    x = x - alpha * grad
    vec.append(np.linalg.norm(2 * A.T @ (A @ x - b)))

plt.semilogy(vec, label='SAG')


### ---------- SAGA ---------- ###
x = np.zeros((m, 1))
gradtable = []

for i in range(n):
    Ai = A[i, :].reshape(1, -1)
    bi = b[i]
    grad_i = 2 * n * Ai.T @ (Ai @ x - bi)
    gradtable.append(grad_i)

gradtable = np.hstack(gradtable)

vec = []
for k in range(1, maxiter + 1):
    alpha = 10 / (16 * L)
    i = np.random.randint(n)
    Ai = A[i, :].reshape(1, -1)
    bi = b[i]
    gradi = 2 * n * Ai.T @ (Ai @ x - bi)
    grad = gradi - gradtable[:, [i]] + (1/n) * np.sum(gradtable, axis=1, keepdims=True)
    gradtable[:, [i]] = gradi
    x = x - alpha * grad
    vec.append(np.linalg.norm(2 * A.T @ (A @ x - b)))

plt.semilogy(vec, label='SAGA')


### ---------- SG---------- ###
x = np.zeros((m, 1))
vec = []

for k in range(1, maxiter + 1):
    alpha = 10 / (16 * L)
    i = np.random.randint(n)
    Ai = A[i, :].reshape(1, -1)
    bi = b[i]
    gradi = 2 * n * Ai.T @ (Ai @ x - bi)
    x = x - alpha * gradi
    vec.append(np.linalg.norm(2 * A.T @ (A @ x - b)))

plt.semilogy(vec, label='SG')

# Final plot formatting (like MATLAB)
plt.xlabel('Iteration')
plt.ylabel('Norm of Gradient')
plt.title('Comparison of SAG, SAGA, and SG')
plt.grid(True)
plt.legend()
plt.show()