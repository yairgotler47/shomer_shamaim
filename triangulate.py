import numpy as np
from scipy import linalg
import cv2
# object position


def DLT(P1, P2, point1, point2):
    A = [
        point1[1] * P1[2, :] - P1[1, :],
        P1[0, :] - point1[0] * P1[2, :],
        point2[1] * P2[2, :] - P2[1, :],
        P2[0, :] - point2[0] * P2[2, :]
    ]
    A = np.array(A).reshape((4, 4))

    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)

    p3d = Vh[3, 0:3] / Vh[3, 3]
    p3d[1] = -p3d[1]  # Flip Y so +Y is UP
    print('Triangulated point: ', p3d)
    return p3d


def triangulate(mtx1, mtx2, image_points1, image_points2, R, T):
    image_points1 = np.array(image_points1)
    image_points2 = np.array(image_points2)
    # RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    P1 = mtx1 @ RT1  # projection matrix for C1

    # RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis=-1)
    P2 = mtx2 @ RT2  # projection matrix for C2

    p3ds = []
    for i in range(len(image_points1)):
        p3ds.append(DLT(P1, P2, image_points1[i], image_points2[i]))
    return p3ds


def triangulate_single(mtx1, mtx2,dist1,dist2, image_point1, image_point2, R, T):
    # RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    P1 = mtx1 @ RT1  # projection matrix for C1

    # RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis=-1)
    P2 = mtx2 @ RT2  # projection matrix for C2

    p1 = np.array([[image_point1]])
    p2 = np.array([[image_point2]])

    undist1=cv2.undistortPoints(p1,mtx1,dist1,P=mtx1)[0][0]
    undist2=cv2.undistortPoints(p2,mtx2,dist2,P=mtx2)[0][0]
    p3d = DLT(P1, P2, undist1, undist2)

    return p3d

