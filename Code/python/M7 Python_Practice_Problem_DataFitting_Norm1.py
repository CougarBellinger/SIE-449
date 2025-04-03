import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

d = 1
n = 100
xbar = np.random.rand(d)
s = np.random.rand(n) * 10 - 2
t = s * xbar + np.random.randn(n)
S = np.column_stack((s, np.ones(n)))

plt.plot(s, t, '*')  # plot dataset


maxiter = 200

def objective_function(x):
    return np.linalg.norm(S@x - t, 1)

fvec = []  # to save objective values in a list and then plot
x = np.zeros(d + 1)  # initialization
x_best = x

for k in range(1, maxiter + 1):
    step_size = 0.02 / np.sqrt(k)  # diminishing stepsize
    subgrad = S.T @ np.sign(S@x-t) # compute subgradient
    x = x - step_size * subgrad  # subgradient step
    if objective_function(x) < objective_function(x_best):  # save x_best
        x_best = x
    fvec.append(objective_function(x)) #save objective values in a vector

vec = np.linspace(-2, 8)
plt.plot(vec, vec * x_best[0] + x_best[1])  # plot the line


plt.figure()
plt.plot(fvec)  # plot the function values

