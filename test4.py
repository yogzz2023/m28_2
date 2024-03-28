import numpy as np
import matplotlib.pyplot as plt

# Define Kalman Filter parameters
dt = 1.0  # Time step
A = np.array([[1, dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])  # State transition matrix (constant velocity)
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])  # Measurement matrix
Q = 0.01 * np.eye(4)  # Process noise covariance
R = 0.1 * np.eye(3)  # Measurement noise covariance

# Initial state estimate
x = np.array([[0], [0], [0], [0]])
P = np.eye(4)  # Initial state covariance

# Placeholder for storing estimated states
estimated_states = []

# Simulated measurement data (replace with your actual measurement data)
measurements = np.random.randn(100, 3)  # 100 measurements of range, azimuth, elevation

# Kalman Filter
for z in measurements:
    # Predict
    x = np.dot(A, x)
    P = np.dot(np.dot(A, P), A.T) + Q

    # Update
    S = np.dot(np.dot(H, P), H.T) + R
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    y = z.reshape(-1, 1) - np.dot(H, x)
    x += np.dot(K, y)
    P = np.dot((np.eye(4) - np.dot(K, H)), P)

    estimated_states.append(x)

estimated_states = np.array(estimated_states)

# Plotting
plt.figure(figsize=(10, 6))

# Plot measured data
plt.scatter(measurements[:, 1], measurements[:, 0], label='Measured', color='blue')

# Plot estimated data
plt.plot(estimated_states[:, 2], estimated_states[:, 0], label='Estimated', color='red')

plt.xlabel('Azimuth')
plt.ylabel('Range')
plt.title('Kalman Filter: Range vs. Azimuth')
plt.legend()
plt.grid(True)
plt.show()
