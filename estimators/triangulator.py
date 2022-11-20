import numpy as np
from utils.geometry import skewsymm_batch


def Triangulation_RANSAC(uv1, P, ransac_n_iter=50, threshold=1e-2):
    A = np.matmul(skewsymm_batch(uv1), P)
    N = A.shape[0]

    best_inlier = 0
    best_C1_X = None
    best_inlier_ids = []
    best_outlier_ids = []
    for i in range(ransac_n_iter):
        rand_idx = np.random.choice(N, size=3, replace=False)
        A_select = A[rand_idx]
        A_select = np.reshape(A_select, (A_select.shape[0] * A_select.shape[1], A_select.shape[2]))
        u, s, vh = np.linalg.svd(A_select)
        C1_X = vh[3, :3] / vh[3, 3]

        proj = (P[:, :, :3] @ C1_X.reshape(3, 1) + P[:, :, 3:]).squeeze()
        proj_n = proj[:, 0:2] / proj[:, 2:3]
        error = np.linalg.norm(proj_n-uv1[:, 0:2], axis=1)
        inlier_ids = np.nonzero(error < threshold)[0]
        inlier_num = inlier_ids.size
        outlier_ids = np.nonzero(error >= threshold)[0]

        if inlier_num > best_inlier:
            best_inlier = inlier_num
            best_inlier_ids = inlier_ids
            best_outlier_ids = outlier_ids
            best_C1_X = C1_X

    return best_C1_X, best_inlier_ids, best_outlier_ids


def Triangulation_LS(uv1_inlier, P_inlier):
    A = np.matmul(skewsymm_batch(uv1_inlier), P_inlier)
    A = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
    u, s, vh = np.linalg.svd(A)
    C1_X = vh[3, :3] / vh[3, 3]

    return C1_X