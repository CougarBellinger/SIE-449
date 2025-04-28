import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time
import scipy.sparse as sp

def sgd(A, b, n, m):
    x = np.zeros(m)
    num_iterations = 10 * n
    alpha_0= 0.1
    x_history = [np.copy(x)]
    epochs=10
    iterations_per_epoch = n
    norm_diff_history = []


    plot_x_axis = np.arange(1, epochs + 1)
    plot_y_axis = []

    batch_size=1
    step=0

    print("Running SGD...")

    start_time = time.time()
    for k in range(1, num_iterations + 1):
        # Choose a random index i
        i = np.random.randint(n)
        a_i = A[i].toarray().flatten() # Get the i-th row of A and flatten
        b_i = b[i]

        # Calculate the gradient of the loss function for the chosen data point
        exponent = -b_i * np.dot(a_i, x)
        gradient_loss = -b_i * (np.exp(exponent) / (1 + np.exp(exponent))) * a_i

        # Calculate the step size
        alpha_k = alpha_0 / np.sqrt(k)

        # Update x using the SGD rule (without proximal operator for now)
        x_prev = np.copy(x)
        x = x - alpha_k * gradient_loss
        x_history.append(np.copy(x))
        norm_diff_history.append(np.linalg.norm(x - x_prev) / alpha_k)

        # Record values for plotting at the end of each epoch
        if k % iterations_per_epoch == 0:
            epoch = k // iterations_per_epoch
            avg_norm_diff = np.mean(norm_diff_history[-iterations_per_epoch:])
            plot_y_axis.append(avg_norm_diff)

    end_time = time.time()
    running_time = end_time - start_time

    print(f"Running time for SGD loop: {running_time:.4f} seconds")

    plt.figure()
    plt.plot(plot_x_axis, plot_y_axis)
    plt.xlabel("Epoch")
    plt.ylabel("||x_k - x_{k-1}|| / alpha_k (Average per Epoch)")
    plt.title("Convergence of SGD")
    plt.grid(True)
    plt.show()

    # UNCOMMENT TO SAVE OUTPUT OF PLOT
    # plt.savefig("Code/hw5/SGD_plot.pdf")

    return (f"Running time for SGD loop: {running_time:.4f} seconds")



def saga(A, b, n, m, lam, L):
    x = np.zeros(m)
    num_iterations = 10 * n
    alpha = 1 / (3 * L)
    x_history = [np.copy(x)]
    epochs = 10
    iterations_per_epoch = n
    norm_diff_history = []
    gradient_history = np.zeros((n, m))  # Store past gradients
    avg_gradient = np.zeros(m)

    plot_x_axis = np.arange(1, epochs + 1)
    plot_y_axis = []

    print("Running SAGA...")
    # Initialize gradient history and average gradient
    for i in range(n):
        a_i = A[i].toarray().flatten()
        b_i = b[i]
        exponent = -b_i * np.dot(a_i, x)
        gradient_history[i] = -b_i * (np.exp(exponent) / (1 + np.exp(exponent))) * a_i
    avg_gradient = np.mean(gradient_history, axis=0)

    start_time = time.time()
    for k in range(1, num_iterations + 1):
        # Choose a random index i
        i = np.random.randint(n)
        a_i = A[i].toarray().flatten()
        b_i = b[i]

        # Calculate the current gradient
        exponent_current = -b_i * np.dot(a_i, x)
        current_gradient = -b_i * (np.exp(exponent_current) / (1 + np.exp(exponent_current))) * a_i

        # Calculate the SAGA descent direction
        descent_direction = current_gradient - gradient_history[i] + avg_gradient

        # Perform the proximal update step
        x_prev = np.copy(x)
        v = x - alpha * descent_direction
        gamma = lam * alpha
        x = np.sign(v) * np.maximum(np.abs(v) - gamma, 0)

        x_history.append(np.copy(x))
        norm_diff_history.append(np.linalg.norm(x - x_prev) / alpha)

        # Update the gradient history and average gradient
        gradient_history[i] = current_gradient
        avg_gradient = np.mean(gradient_history, axis=0)

        # Record values for plotting at the end of each epoch
        if k % iterations_per_epoch == 0:
            epoch = k // iterations_per_epoch
            avg_norm_diff = np.mean(norm_diff_history[-iterations_per_epoch:])
            plot_y_axis.append(avg_norm_diff)

    end_time = time.time()
    running_time = end_time - start_time

    plt.figure()
    plt.plot(plot_x_axis, plot_y_axis)
    plt.xlabel("Epoch")
    plt.ylabel("||x_k - x_{k-1}|| / alpha (Average per Epoch)")
    plt.title("Convergence of Proximal SAGA")
    plt.grid(True)
    plt.show()

    # UNCOMMENT TO SAVE OUTPUT OF PLOT
    # plt.savefig("Code/hw5/SAGA_plot.pdf")

    print(f"Running time for Proximal SAGA loop: {running_time:.4f} seconds")
    return (f"Running time for Proximal SAGA loop: {running_time:.4f} seconds")

def svrgPlusPlus(A, b, n, m, lam, L, outer_iterations=10, inner_iterations=None):
    if inner_iterations is None:
        inner_iterations = n

    x = np.zeros(m)
    tilde_x = np.zeros(m)  # The snapshot vector
    alpha = 1 / (3 * L)
    x_history = [np.copy(x)]
    norm_diff_history = []
    plot_x_axis = np.arange(1, outer_iterations + 1)
    plot_y_axis = []

    print("Running SVRG++...")

    start_time = time.time()
    for s in range(outer_iterations):
        tilde_x = np.copy(x)
        full_grad = np.zeros(m)
        for i in range(n):
            a_i = A[i].toarray().flatten()
            b_i = b[i]
            exponent = -b_i * np.dot(a_i, tilde_x)
            full_grad += -b_i * (np.exp(exponent) / (1 + np.exp(exponent))) * a_i
        full_grad /= n

        x_prev_outer = np.copy(x)
        for t in range(inner_iterations):
            i = np.random.randint(n)
            a_i = A[i].toarray().flatten()
            b_i = b[i]

            # Gradient at current x
            exponent_x = -b_i * np.dot(a_i, x)
            grad_x_i = -b_i * (np.exp(exponent_x) / (1 + np.exp(exponent_x))) * a_i

            # Gradient at snapshot tilde_x
            exponent_tilde_x = -b_i * np.dot(a_i, tilde_x)
            grad_tilde_x_i = -b_i * (np.exp(exponent_tilde_x) / (1 + np.exp(exponent_tilde_x))) * a_i

            # SVRG++ gradient update (a common form)
            grad_estimate = grad_x_i - grad_tilde_x_i + full_grad

            # Proximal update
            v = x - alpha * grad_estimate
            gamma = lam * alpha
            x = np.sign(v) * np.maximum(np.abs(v) - gamma, 0)
            x_history.append(np.copy(x))

        norm_diff_history.append(np.linalg.norm(x - x_prev_outer) / alpha)
        plot_y_axis.append(norm_diff_history[-1]) # Store the norm difference at the end of each outer loop

    end_time = time.time()
    running_time = end_time - start_time


    plt.figure()
    plt.plot(plot_x_axis, plot_y_axis)
    plt.xlabel("Outer Iteration (Epoch)")
    plt.ylabel("||x_k - x_{k-1}|| / alpha (Last iterates of inner loop)")
    plt.title("Convergence of Proximal SVRG++")
    plt.grid(True)
    plt.show()

    # UNCOMMENT TO SAVE OUTPUT OF PLOT
    # plt.savefig("Code/hw5/SVRG++_plot.pdf")

    print(f"Running time for Proximal SVRG++ loop: {running_time:.4f} seconds")
    return (f"Running time for Proximal SVRG++ loop: {running_time:.4f} seconds")

def main():
    np.random.seed(123)

    data = scipy.io.loadmat('Code/hw5/mushrooms.mat')
    A = data['A']
    b = data['b']

    n, m = A.shape
    lam = 1e-5;
    L_max = 0

    for i in range(n):
        L_i = np.linalg.norm(A[i].toarray())**2 / 4
        L_max = max(L_max, L_i)

    print(f"L_max is: {L_max:.4f}")


    sgd(A=A, b=b, n=n,m=m)
    saga(A=A, b=b, n=n,m=m, lam=lam, L=L_max)
    svrgPlusPlus(A=A, b=b, n=n, m=m, lam=lam, L=L_max, outer_iterations=10, inner_iterations=n)

    # UNCOMMENT TO WRITE OUTPUT TO TXT
    # outputFile = open("Code/hw5/hw5_output.txt", "w")
    # outputFile.write(sgd(A=A, b=b, n=n,m=m))
    # outputFile.write(saga(A=A, b=b, n=n,m=m, lam=lam, L=L_max))
    # outputFile.write(svrgPlusPlus(A=A, b=b, n=n, m=m, lam=lam, L=L_max, outer_iterations=10, inner_iterations=n))
    # outputFile.close()

if __name__ == "__main__":
    main()