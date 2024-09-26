import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
data = pd.read_csv('../data/Advertising_N200_p3.csv')
X = data[['TV', 'radio', 'newspaper']].values
y = data['sales'].values  

N = len(y)  # observations
p = X.shape[1]  
X_design = np.hstack((np.ones((N, 1)), X)) 

# Initialize parameters
beta = np.random.uniform(-1, 1, p + 1)  

# hyperparameters
alpha = 2.5e-6  # Learning rate
batch_size = 10  # Mini-batch size
num_iterations = 20000  # Number of iterations

betas_over_time = np.zeros((num_iterations, p + 1))
costs = np.zeros(num_iterations)

# Mini-batch gradient descent
for iteration in range(num_iterations):
    indices = np.random.permutation(N)
    X_shuffled = X_design[indices]
    y_shuffled = y[indices]

    for b in range(0, N, batch_size):
        X_batch = X_shuffled[b:b + batch_size]
        y_batch = y_shuffled[b:b + batch_size]

        error = y_batch - X_batch @ beta
        gradient = -2 * (X_batch.T @ error) / batch_size
        beta -= alpha * gradient

    betas_over_time[iteration] = beta
    costs[iteration] = np.sum((y - X_design @ beta) ** 2) / N

# Deliverable 1 
plt.figure(figsize=(10, 6))
for j in range(p + 1):
    plt.plot(range(num_iterations), betas_over_time[:, j], label=f'β_{j}')
plt.xlabel('Iteration')
plt.ylabel('Coefficient Value')
plt.title('Effect of Iteration on Regression Coefficients')
plt.legend()
plt.show()

# Deliverable 2 
plt.figure(figsize=(10, 6))
plt.plot(range(num_iterations), costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Effect of Iteration on Cost Function')
plt.show()

# Deliverable 3
print("Estimated parameters (beta):", beta)

# Deliverable 4 
predictions = X_design @ beta
mse = np.mean((y - predictions) ** 2)
print("Mean Squared Error (MSE):", mse)
