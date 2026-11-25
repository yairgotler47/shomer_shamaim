import numpy as np
import matplotlib.pyplot as plt

GRAVITY_CM_S2 = 980.0


def init_filter_matrices():
    H = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ])

    Q = np.diag([1.0, 1.0, 1.0, 15.0, 15.0, 15.0])

    # Measurement noise (cm^2)
    R = np.eye(3) * 4.0

    # Initial state uncertainty
    P = np.diag([4.0, 4.0, 4.0, 10000.0, 10000.0, 10000.0])

    return H, Q, R, P


def build_motion_matrices(dt, g=GRAVITY_CM_S2):
    F = np.array([
        [1.0, 0.0, 0.0, dt,  0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, dt,  0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, dt],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])

    B_u = np.array([0.0, -0.5 * g * (dt ** 2), 0.0, 0.0, -g * dt, 0.0])

    return F, B_u


def predict(state, P_mat, B_u, F, Q):
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


def predict_trajectory(state, P, Q, dt=0.033, steps=20):
    F, B_u = build_motion_matrices(dt)
    rollout_state = state.copy()
    rollout_P = P.copy()

    future_pts = []
    for _ in range(steps):
        rollout_state, rollout_P = predict(rollout_state, rollout_P, B_u, F, Q)
        future_pts.append((rollout_state[0], rollout_state[1], rollout_state[2]))

    return future_pts


def show_3d_graph(points3d_1, points3d_2):
    if not points3d_1 and not points3d_2:
        return

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    if points3d_1:
        x_1, y_1, z_1 = zip(*points3d_1)
        ax.scatter(x_1, y_1, z_1, c="red", label="Observed")

    if points3d_2:
        x_2, y_2, z_2 = zip(*points3d_2)
        ax.scatter(x_2, y_2, z_2, c="green", label="Predicted")

    ax.set_title("3D Trajectory (cm)")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    ax.legend()
    plt.show()

def calc_rotation(point3d,translation_vec):
    curr_point=[point3d[0]+translation_vec[0],point3d[1]+translation_vec[1],point3d[1]+translation_vec[1]]
    lat=np.arctan(curr_point[1]/curr_point[2])
    pitch=np.arctan(curr_point[0]/curr_point[2])
    return lat,pitch