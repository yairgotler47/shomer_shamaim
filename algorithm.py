import cv2
import numpy as np

height, width = 800, 1200
base_img = np.zeros((height, width, 3), dtype=np.uint8)

# Pepsi
dt = 0.033
g = 9.8
v0 = 50
theta0 = np.radians(50.0)

vx = v0 * np.cos(theta0)
vy = v0 * np.sin(theta0)
x, y = 30.0, 50.0

gt_x, gt_y = [x], [y]
while True:
    x += vx * dt
    y += vy * dt - 0.5 * g * (dt ** 2)
    vy -= g * dt
    if y < 0:
        break
    gt_x.append(x)
    gt_y.append(y)

gt_x = np.array(gt_x)
gt_y = np.array(gt_y)
n_steps = len(gt_x)

# Kalman Filter Matrices
F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

B_u = np.array([0, -0.5 * g * (dt ** 2), 0, -g * dt])
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])


# Process noise
Q = np.eye(4) * 0.05
# Measurement noise
R = np.eye(2) * 4.0
# Initial uncertainty
P = np.diag([4.0, 4.0, 400.0, 400.0])

def predict(state, P_mat):
    state_pred = F @ state + B_u
    P_pred = F @ P_mat @ F.T + Q
    return state_pred, P_pred


def update(state_pred, P_pred, measurement):
    y_err = measurement - H @ state_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    state_upd = state_pred + K @ y_err
    P_upd = (np.eye(4) - K @ H) @ P_pred
    return state_upd, P_upd


def draw_covariance_ellipse(img, center, cov_2x2, scale=2.0, color=(255, 255, 0)):
    # Eigenvalues give axis lengths squared; eigenvectors give orientation
    vals, vecs = np.linalg.eigh(cov_2x2)

    vals = np.maximum(vals, 0)

    # Semiaxes lengths
    width_axis = int(scale * np.sqrt(vals[1]))
    height_axis = int(scale * np.sqrt(vals[0]))

    # Angle of rotation in degrees
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))

    cv2.ellipse(
        img,
        (int(center[0]), int(center[1])),
        (max(1, width_axis), max(1, height_axis)),
        -angle,
        0,
        360,
        color,
        1,
    )


# State: [x, y, vx, vy]
state = np.array([gt_x[0], gt_y[0], 0.0, 0.0])
min_observed_steps = 2

for step in range(n_steps):
    meas = np.array(
        [gt_x[step] + np.random.normal(0, 3.0),
         gt_y[step] + np.random.normal(0, 3.0)]
    )

    # Filter Predict & Update
    state_pred, P_pred = predict(state, P)
    state, P = update(state_pred, P_pred, meas)

    frame = base_img.copy()

    # Draw past ground truth history
    for i in range(step + 1):
        px = int(gt_x[i])
        py = int(height - gt_y[i])
        cv2.circle(frame, (px, py), 2, (180, 50, 0), -1)

    # Current estimated position
    cur_px = int(state[0])
    cur_py = int(height - state[1])
    cv2.circle(frame, (cur_px, cur_py), 5, (0, 255, 0), -1)

    # Trajectory prediction
    if step >= min_observed_steps:
        rollout_state = state.copy()
        rollout_P = P.copy()

        future_pts = []

        for future_i in range(250):
            rollout_state = F @ rollout_state + B_u
            rollout_P = F @ rollout_P @ F.T + Q

            if rollout_state[1] < 0:
                break

            fx = rollout_state[0]
            fy = height - rollout_state[1]
            future_pts.append((fx, fy))

            # Draw an error ellipse every 10 steps into the future
            if future_i % 10 == 0 and future_i > 0:
                pos_cov = rollout_P[:2, :2]
                draw_covariance_ellipse(
                    frame, (fx, fy), pos_cov, scale=2.0, color=(255, 200, 0)
                )

        # Draw predicted trajectory centerline (Red)
        for pt in future_pts:
            px, py = int(pt[0]), int(pt[1])
            if 0 <= px < width and 0 <= py < height:
                cv2.circle(frame, (px, py), 1, (0, 0, 255), -1)

    cv2.imshow("Trajectory and Uncertainty Propagation", frame)
    if cv2.waitKey(70) & 0xFF == 27:
        break

cv2.destroyAllWindows()
