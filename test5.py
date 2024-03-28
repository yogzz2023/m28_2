import numpy as np
import matplotlib.pyplot as plt

# Define Kalman filter parameters
# Define state transition matrix
A = np.eye(4)  # Assuming a simple model where state is [range, azimuth, elevation, time]
# Define measurement matrix
H = np.eye(4)
# Define process noise covariance
Q = np.eye(4) * 0.01
# Define measurement noise covariance
R = np.eye(4) * 0.1

# Initialize state estimate
x = np.array([initial_range_estimate, initial_azimuth_estimate, initial_elevation_estimate, initial_time_estimate]).reshape(-1, 1)
# Initialize covariance matrix
P = np.eye(4) * 0.1

# Kalman filter prediction and update steps
def kalman_filter(z):
    global x, P
    # Prediction
    x_pred = np.dot(A, x)
    P_pred = np.dot(A, np.dot(P, A.T)) + Q

    # Update
    y = z - np.dot(H, x_pred)
    S = np.dot(H, np.dot(P_pred, H.T)) + R
    K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(S)))

    x = x_pred + np.dot(K, y)
    P = P_pred - np.dot(K, np.dot(H, P_pred))

    return x, P

# Joint Probabilistic Data Association (JPDA)
def jpda_measurement_association(measurements, predicted_states):
    num_predicted_states = predicted_states.shape[0]
    num_measurements = measurements.shape[0]
    association_probabilities = np.zeros((num_predicted_states, num_measurements))

    # Calculate measurement likelihoods
    measurement_likelihoods = np.zeros((num_predicted_states, num_measurements))
    for i in range(num_predicted_states):
        for j in range(num_measurements):
            measurement_likelihoods[i, j] = calculate_measurement_likelihood(predicted_states[i], measurements[j], R)

    # Compute association probabilities
    for j in range(num_measurements):
        likelihood_sum = np.sum(measurement_likelihoods[:, j])
        if likelihood_sum > 0:
            association_probabilities[:, j] = measurement_likelihoods[:, j] / likelihood_sum

    # Assign measurements to predicted states based on association probabilities
    associations = np.argmax(association_probabilities, axis=0)

    return associations

def calculate_measurement_likelihood(predicted_state, measurement, measurement_noise_covariance):
    residual = measurement - predicted_state
    covariance_inverse = np.linalg.inv(measurement_noise_covariance)
    mahalanobis_distance = np.sqrt(np.dot(residual.T, np.dot(covariance_inverse, residual)))
    likelihood = np.exp(-0.5 * mahalanobis_distance)
    return likelihood

# Sample data for measurements (replace with actual measurements)
num_measurements = 100
measurements = np.random.randn(num_measurements, 4)  # Assuming each measurement is [range, azimuth, elevation, time]

# Perform Kalman filtering and JPDA
for measurement in measurements:
    # Perform Kalman filtering
    x, P = kalman_filter(measurement)

    # Perform JPDA
    associations = jpda_measurement_association(measurement, x)

    # Use the updated states for prediction and plotting
    predicted_range = x[0]
    predicted_azimuth = x[1]
    predicted_elevation = x[2]
    predicted_time = x[3]

    # Plot graphs for range vs time, azimuth vs time, elevation vs time
    # Compare predicted and measured values

# Plotting code for comparison
plt.plot(predicted_range, predicted_time, label='Predicted Range')
plt.plot(measured_range, measured_time, 'o', label='Measured Range')
plt.xlabel('Time')
plt.ylabel('Range')
plt.legend()
plt.show()

# Repeat plotting for azimuth and elevation
