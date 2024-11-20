import numpy as np
import matplotlib.pyplot as plt
from Activation import activate

# Training Data
xt = np.array([[0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7],
               [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]])
yt = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])

# Initialize the weights and biases
np.random.seed(5000)
W2 = 0.5 * np.random.randn(2, 2)
W3 = 0.5 * np.random.randn(3, 2)
W4 = 0.5 * np.random.randn(2, 3)
b2 = 0.5 * np.random.randn(2, 1)
b3 = 0.5 * np.random.randn(3, 1)
b4 = 0.5 * np.random.randn(2, 1)

eta = 0.05  # Set learning rate
Niter = int(1e6)  # Set Max Iterations
savecost = np.zeros(Niter)  # Stores cost values

# Cost function
def cost(W2, W3, W4, b2, b3, b4):
    total_cost = 0
    for j in range(10):
        a2 = activate(xt[:, j:j+1], W2, b2)
        a3 = activate(a2, W3, b3)
        a4 = activate(a3, W4, b4)
        total_cost = total_cost + np.linalg.norm(yt[:, j:j+1] - a4) ** 2
    return total_cost
    
# Doing Backprogagation
for i in range(Niter):
    k = np.random.randint(10)
    x = xt[:, k:k+1]
    y = yt[:, k:k+1]

    # Forward
    a2 = activate(x, W2, b2)
    a3 = activate(a2, W3, b3)
    a4 = activate(a3, W4, b4)

    # Backward 
    delta4 = a4 * (1 - a4) * (a4 - y)
    delta3 = a3 * (1 - a3) * (W4.T @ delta4)
    delta2 = a2 * (1 - a2) * (W3.T @ delta3)

    # Descent step
    W2 = W2- eta * delta2 @ x.T
    W3 = W3 - eta * delta3 @ a2.T
    W4 = W4 - eta * delta4 @ a3.T
    b2 = b2 - eta * delta2
    b3 = b3 - eta * delta3
    b4 = b4 - eta * delta4

    # Monitor and display cost
    newcost = cost(W2, W3, W4, b2, b3, b4)
    savecost[i] = newcost
    if i % 1000 == 0:
        print(f"Iteration {i}: Cost = {newcost}")
    
    #Save Parameters
    np.savez("model_parameters.npz", W2=W2, W3=W3, W4=W4, b2=b2, b3=b3, b4=b4)
    np.save("cost_vec.npy", savecost)

# Plot the cost decay every 1000 iterations
plt.semilogy(range(0, Niter, 1000), savecost[0:Niter:1000])
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Decay Over Iterations")
plt.show()
