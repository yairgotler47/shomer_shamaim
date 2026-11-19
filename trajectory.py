import cv2
import numpy as np
import matplotlib.pyplot as plt

def define_matrices():
    dt = 0.033
    g = 980
    # Kalman Filter Matrices
    F = np.array([[1, 0, 0, dt, 0, 0], [0, 1, 0, 0, dt, 0], [0, 0, 1, 0, 0, dt],
              [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])

    B_u = np.array([0, -0.5 * g * (dt ** 2), 0, 0, -g * dt, 0])
    H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])


    # Process noise
    Q = np.eye(6) * 0.05
    # Measurement noise
    R = np.eye(3) * 4.0
    # Initial uncertainty
    P = np.diag([4.0, 4.0, 4.0, 400.0, 400.0, 400.0])

    return F,B_u,H,Q,R,P

def predict(state, P_mat,B_u, F ,Q):
    state_pred = F @ state + B_u
    P_pred = F @ P_mat @ F.T + Q
    return state_pred, P_pred


def update(state_pred, P_pred, H, R, measurement):
    y_err = measurement - H @ state_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    state_upd = state_pred + K @ y_err
    P_upd = (np.eye(6) - K @ H) @ P_pred
    return state_upd, P_upd



def predict_trajectory(state, P , F , B_u, Q):
    rollout_state = state.copy()
    rollout_P = P.copy()

    future_pts = []

    for future_i in range(60):
        rollout_state = F @ rollout_state + B_u
        rollout_P = F @ rollout_P @ F.T + Q


        fx = rollout_state[0]
        fy = rollout_state[1]
        fz = rollout_state[2]
        future_pts.append((fx, fy, fz))

    return future_pts

def show_3d_graph(points3d_1, points3d_2):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x_1, y_1, z_1 = zip(*points3d_1)
    ax.scatter(x_1, y_1, z_1, c='red')

    x_2, y_2, z_2 = zip(*points3d_2)
    ax.scatter(x_2, y_2, z_2, c='green')

    ax.set_title('3D Scatter Plot')
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    plt.show()
    return
