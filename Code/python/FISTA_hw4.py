import numpy as np

np.random.seed(123)

n = 500
m = 100
x = np.zeros((n, 1))
A = np.random.randn(m, n)
lmbda = 10**-3
maxiter = 200

y = x.copy()
t = 1

for k in range(1, maxiter + 1):
    L = np.linalg.norm(A.T @ A) / 4
    alpha = 1 / L

    z = -A @ y
    exp_z = np.exp(z)
    grad = -A.T @ (exp_z / (1 + exp_z))

    x_new = y - alpha * grad
    x_next = np.maximum(np.abs(x_new) - alpha * lmbda, 0) * np.sign(x_new)

    t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
    y = x_next + (t - 1) / t_new * (x_next - x)

    x = x_next.copy()
    t = t_new

# Calculate the objective value
z = -A @ x
objective_value = np.sum(np.log(1 + np.exp(z))) + lmbda * np.linalg.norm(x, 1)

# Count the number of zeros
num_zeros = np.sum(x == 0)

print(f'Objective value from FISTA: {objective_value}')
print(f'Number of zeros in x: {num_zeros}')