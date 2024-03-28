import numpy as np
import matplotlib.pyplot as plt

# Kalman Filter functions
def predict(x, P, F, Q):
    x_pred = np.dot(F, x)
    P_pred = np.dot(np.dot(F, P), F.T) + Q
    return x_pred, P_pred

def update(x_pred, P_pred, z, H, R):
    y = z - np.dot(H, x_pred)
    S = np.dot(np.dot(H, P_pred), H.T) + R
    K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(S))
    x_updated = x_pred + np.dot(K, y)
    P_updated = P_pred - np.dot(np.dot(K, H), P_pred)
    return x_updated, P_updated, y, K, S

# Joint Probabilistic Data Association (JPDA)
def measurement_log_likelihood(z, x_pred, P_pred, H, R):
    innovation = z - np.dot(H, x_pred)
    S = np.dot(np.dot(H, P_pred), H.T) + R
    log_det_S = np.log(np.linalg.det(S))
    log_likelihood = -0.5 * (len(z) * np.log(2 * np.pi) + log_det_S + np.dot(innovation.T, np.dot(np.linalg.inv(S), innovation)))
    return log_likelihood

def association_probabilities(z, x_pred, P_pred, H, R, measurements):
    log_likelihoods = []
    for measurement in measurements:
        log_likelihood = measurement_log_likelihood(measurement, x_pred, P_pred, H, R)
        log_likelihoods.append(log_likelihood)
    max_log_likelihood = max(log_likelihoods)
    exp_log_likelihoods = np.exp(log_likelihoods - max_log_likelihood) # Subtract max for numerical stability
    association_probs = exp_log_likelihoods / np.sum(exp_log_likelihoods)
    return association_probs

# Constants
dt = 1.0  # Time step
F = np.array([[1, dt, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, dt, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, dt, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, dt, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1]])  # State transition matrix
H = np.eye(9)  # Measurement matrix
Q = np.eye(9) * 0.01  # Process noise covariance
R = np.eye(9) * 0.1  # Measurement noise covariance

# Initial state for target 1
x_1 = np.array([0, 10, 0, 5, 0, 10, 0, 5, 0])  # [position_x, velocity_x, position_y, velocity_y, position_z, velocity_z]
P_1 = np.eye(9)  # Initial state covariance for target 1

# Measurements for target 1
measurements = [
    np.array([94779.54, 217.0574, 2.7189, 21486.916, 0, 0, 0, 0, 0]),
    np.array([27197.81, 153.2595, 1.2913, 21487.193, 0, 0, 0, 0, 0]),
    np.array([85839.11, 226.6049, 5.0573, 21487.252, 0, 0, 0, 0, 0])
]

# Constants
dt = 1.0  # Time step
time_steps = np.arange(1, len(measurements) + 1) * dt  # Time steps

# Extracting measurement and predicted values
ranges = np.array([z[0] for z in measurements])
azimuths = np.array([z[1] for z in measurements])
elevations = np.array([z[2] for z in measurements])
predicted_ranges_kalman = []
predicted_azimuths_kalman = []
predicted_elevations_kalman = []
predicted_ranges_jpda = []
predicted_azimuths_jpda = []
predicted_elevations_jpda = []

# Kalman Filter predictions
x_1 = np.array([0, 10, 0, 5, 0, 10, 0, 5, 0])  # Reset initial state for Kalman Filter
for z in measurements:
    # Predict step
    x_pred, _ = predict(x_1, P_1, F, Q)
    predicted_ranges_kalman.append(x_pred[0])
    predicted_azimuths_kalman.append(x_pred[1])
    predicted_elevations_kalman.append(x_pred[2])
    x_1 = x_pred  # Update state for next iteration

# JPDA predictions
x_1 = np.array([0, 10, 0, 5, 0, 10, 0, 5, 0])  # Reset initial state for JPDA
for z in measurements:
    # Predict step
    x_pred, _ = predict(x_1, P_1, F, Q)
    association_probs = association_probabilities(z, x_pred, P_1, H, R, measurements)
    x_pred_weighted = np.zeros_like(x_pred)
    for i, prob in enumerate(association_probs):
        x_pred_weighted += prob * x_pred
    predicted_ranges_jpda.append(x_pred_weighted[0])
    predicted_azimuths_jpda.append(x_pred_weighted[1])
    predicted_elevations_jpda.append(x_pred_weighted[2])
    x_1 = x_pred  # Update state for next iteration

# Plotting
plt.figure(figsize=(12, 8))

# Range vs Time
plt.subplot(3, 1, 1)
plt.plot(time_steps, ranges, marker='o', linestyle='-', color='blue', label='Measurement (Range)')
plt.plot(time_steps, predicted_ranges_kalman, marker='s', linestyle='--', color='red', label='Predicted (Kalman Filter)')
plt.plot(time_steps, predicted_ranges_jpda, marker='x', linestyle=':', color='green', label='Predicted (JPDA)')
plt.xlabel('Time (s)')
plt.ylabel('Range')
plt.title('Range vs Time')
plt.legend()

# Azimuth vs Time
plt.subplot(3, 1, 2)
plt.plot(time_steps, azimuths, marker='o', linestyle='-', color='blue', label='Measurement (Azimuth)')
plt.plot(time_steps, predicted_azimuths_kalman, marker='s', linestyle='--', color='red', label='Predicted (Kalman Filter)')
plt.plot(time_steps, predicted_azimuths_jpda, marker='x', linestyle=':', color='green', label='Predicted (JPDA)')
plt.xlabel('Time (s)')
plt.ylabel('Azimuth')
plt.title('Azimuth vs Time')
plt.legend()

# Elevation vs Time
plt.subplot(3, 1, 3)
plt.plot(time_steps, elevations, marker='o', linestyle='-', color='blue', label='Measurement (Elevation)')
plt.plot(time_steps, predicted_elevations_kalman, marker='s', linestyle='--', color='red', label='Predicted (Kalman Filter)')
plt.plot(time_steps, predicted_elevations_jpda, marker='x', linestyle=':', color='green', label='Predicted (JPDA)')
plt.xlabel('Time (s)')
plt.ylabel('Elevation')
plt.title('Elevation vs Time')
plt.legend()

plt.tight_layout()
plt.show()
