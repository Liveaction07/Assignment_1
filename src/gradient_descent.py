import numpy as np
import pandas as pd

# Load the dataset
data_path = '../data/Advertising_N200_p3.csv'  # Path to dataset
data = pd.read_csv(data_path)

# Set up feature matrix X and response vector y
X = data[['TV', 'radio', 'newspaper']].values
y = data['sales'].values

# Add a column of ones to X to account for the intercept term (β0)
X = np.c_[np.ones(X.shape[0]), X]  # Adding a column of ones for the intercept term

# Initialize parameters (β0, β1, β2, β3) randomly between -1 and 1
beta = np.random.uniform(-1, 1, X.shape[1])

# Define the learning rate and number of iterations
learning_rate = 2.5e-6
iterations = 20000
batch_size = 10

# Define the cost function
def compute_cost(X, y, beta):
    m = len(y)  # Number of training examples
    predictions = X.dot(beta)  # Predicted values
    cost = (1/m) * np.sum((predictions - y) ** 2)  # Sum of squared errors (SSE)
    return cost

# Define the mini-batch gradient descent function
def mini_batch_gradient_descent(X, y, beta, learning_rate, iterations, batch_size):
    m = len(y)  # Number of observations
    cost_history = []  # List to store cost after each iteration
    
    for it in range(iterations):
        # Shuffle data at the start of each iteration
        idx = np.random.permutation(m)
        X_shuffled = X[idx]
        y_shuffled = y[idx]

        for i in range(0, m, batch_size):
            # Create mini-batch
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            # Compute predictions for the batch
            predictions = X_batch.dot(beta)

            # Calculate the gradient
            gradient = (2 / batch_size) * X_batch.T.dot(predictions - y_batch)

            # Update the parameters
            beta = beta - learning_rate * gradient

        # Calculate and store the cost after each iteration
        cost = compute_cost(X, y, beta)
        cost_history.append(cost)

        # Print cost for every 1000th iteration
        if it % 1000 == 0:
            print(f"Iteration {it}: Cost {cost}")

    return beta, cost_history

# Run mini-batch gradient descent
final_beta, cost_history = mini_batch_gradient_descent(X, y, beta, learning_rate, iterations, batch_size)

# Print final parameters
print("Final parameters after gradient descent:", final_beta)

# Optionally, plot the cost history (to visualize convergence)
import matplotlib.pyplot as plt

plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.show()
