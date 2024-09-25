import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    beta_history = []  # List to store the evolution of beta (coefficients)

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

        # Store the beta values for each iteration
        beta_history.append(beta.copy())

        # Print cost for every 1000th iteration
        if it % 1000 == 0:
            print(f"Iteration {it}: Cost {cost}")

    return beta, cost_history, beta_history

# Run mini-batch gradient descent
final_beta, cost_history, beta_history = mini_batch_gradient_descent(X, y, beta, learning_rate, iterations, batch_size)

# Convert beta_history to a numpy array for easier plotting
beta_history = np.array(beta_history)

# 1. Print Best-Fit Model Parameters
print("\nBest-fit model parameters (Intercept and Coefficients):")
print(f"Intercept: {final_beta[0]:.6f}")
print(f"TV Coefficient: {final_beta[1]:.6f}")
print(f"Radio Coefficient: {final_beta[2]:.6f}")
print(f"Newspaper Coefficient: {final_beta[3]:.6f}")

# 2. Calculate and Print MSE on the training set
def mean_squared_error(X, y, beta):
    predictions = X.dot(beta)  # Predicted values
    mse = np.mean((predictions - y) ** 2)  # Mean squared error
    return mse

mse = mean_squared_error(X, y, final_beta)
print(f"\nMean Squared Error on the training set: {mse:.6f}")

# 3. Plot the cost function over iterations
plt.figure(figsize=(8, 6))
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.show()

# 4. Plot the evolution of beta coefficients over iterations
plt.figure(figsize=(8, 6))
plt.plot(beta_history[:, 0], label='Intercept')
plt.plot(beta_history[:, 1], label='TV Coefficient')
plt.plot(beta_history[:, 2], label='Radio Coefficient')
plt.plot(beta_history[:, 3], label='Newspaper Coefficient')
plt.xlabel('Iterations')
plt.ylabel('Coefficient Value')
plt.title('Evolution of Coefficients over Iterations')
plt.legend()
plt.show()
