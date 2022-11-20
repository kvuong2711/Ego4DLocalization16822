from scipy.optimize import least_squares
from utils.geometry import *


def PnP(f2d, f3d, K2):
    # the minimal number of points accepted by solvePnP is 4:
    ret = cv2.solvePnPRansac(f3d,
                             f2d,
                             K2,
                             distCoeffs=None,
                             flags=cv2.SOLVEPNP_EPNP)
    success = ret[0]
    rotation_vector = ret[1]
    translation_vector = ret[2]

    f_2d = np.linalg.inv(K2) @ np.concatenate((f2d[:, 0],
                                               np.ones((f2d.shape[0], 1))), axis=1).T

    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    translation_vector = translation_vector.reshape(3)
    proj = rotation_mat @ f3d[:, 0].T + translation_vector.reshape(3, -1)
    proj = proj[:2] / proj[2:]
    reproj_error = np.linalg.norm(f_2d[:2] - proj[:2], axis=0)
    reproj_inliers = reproj_error < 1e-2
    reproj_inliers = reproj_inliers.reshape(-1)

    Caz_T_Wmp = np.eye(4)
    Caz_T_Wmp[:3, :3] = rotation_mat
    Caz_T_Wmp[:3, 3] = translation_vector

    if success == 0 or reproj_inliers.sum() < 4:
        return 0, None, None, None
    else:
        return True, Caz_T_Wmp, f2d[reproj_inliers, 0], f3d[reproj_inliers, 0]


def refine_pnp(C_uv, Caz_T_Wmp, G_p_f, K2):
    C_b_f = np.linalg.inv(K2) @ np.concatenate((C_uv, np.ones((1, C_uv.shape[1]))), axis=0)
    C_b_f_norm = np.linalg.norm(C_b_f, axis=0, keepdims=True)
    C_b_f /= C_b_f_norm
    C_T_G_opt = RunPnPNL(Caz_T_Wmp, G_p_f, C_b_f,
                         hm_ids=np.arange(0, G_p_f.shape[1], 1),
                         cutoff=8e-3)
    return C_T_G_opt


def skewsymm(x):
    Sx = np.zeros((3, 3))
    Sx[0, 1] = -x[2]
    Sx[0, 2] = x[1]
    Sx[1, 0] = x[2]
    Sx[2, 0] = -x[1]
    Sx[1, 2] = -x[0]
    Sx[2, 1] = x[0]

    return Sx

def construct_control_point_matrix(G_p_f, C_b_f_m, G_P_ctrl):
    Ncorrespondences = G_p_f.shape[1]

    # Compute alpha coeff
    alphas = np.zeros((4, Ncorrespondences))
    M = np.zeros((3 * Ncorrespondences, 12))

    for i in range(Ncorrespondences):
        A = np.concatenate((G_P_ctrl, np.ones((1, 4))), axis=0)
        b = np.concatenate((G_p_f[:, i], np.ones((1))), axis=0)
        alphas[:, i] = np.linalg.solve(A, b)

        M[(3 * i):(3 * (i + 1)), 0:3] = alphas[0, i] * skewsymm(C_b_f_m[:, i])
        M[(3 * i):(3 * (i + 1)), 3:6] = alphas[1, i] * skewsymm(C_b_f_m[:, i])
        M[(3 * i):(3 * (i + 1)), 6:9] = alphas[2, i] * skewsymm(C_b_f_m[:, i])
        M[(3 * i):(3 * (i + 1)), 9:12] = alphas[3, i] * skewsymm(C_b_f_m[:, i])

    return M


def EPnP_solver(M, G_P_ctrl):
    _, S, Vt = np.linalg.svd(M)

    C_P_ctrl = Vt[-1]
    C_P_ctrl = C_P_ctrl.reshape((4, 3)).T

    U, _, Vh = np.linalg.svd((C_P_ctrl[:, :3] - C_P_ctrl[:, 3:4]) @ np.linalg.inv(G_P_ctrl[:, :3] - G_P_ctrl[:, 3:4]))
    C_R_G_hat = U @ Vh
    C_R_G_hat *= np.linalg.det(C_R_G_hat)

    C_P_ctrl0 = C_R_G_hat @ (G_P_ctrl[:, 0:1] - G_P_ctrl[:, 1:2])
    scale = np.linalg.norm(G_P_ctrl[:, 0] - G_P_ctrl[:, 1]) / np.linalg.norm(C_P_ctrl[:, 0] - C_P_ctrl[:, 1]) \
            * np.sign(C_P_ctrl[0, 0] - C_P_ctrl[0, 1]) * np.sign(C_P_ctrl0[0, 0])

    C_t_G_hat = scale * C_P_ctrl[:, 0:1] - C_R_G_hat @ G_P_ctrl[:, 0:1]

    return C_R_G_hat, C_t_G_hat


def SetupPnPNL(C_T_G, G_p_f, C_b_f):
    n_points = G_p_f.shape[1]
    n_projs = n_points
    b = np.zeros((3 * n_projs,))

    for k in range(n_points):
        b[3 * k: 3 * (k + 1)] = C_b_f[:, k]

    z = np.zeros((7 + 3 * n_points,))
    R = C_T_G[:3, :3]
    t = C_T_G[:3, 3]
    q = Rotation2Quaternion(R)

    z[0:7] = np.concatenate([t, q])
    for i in range(n_points):
        z[7 + 3 * i: 7 + 3 * (i + 1)] = G_p_f[:, i]

    return z, b


def MeasureReprojectionSinglePose(z, b, p, hm_ids):

    n_projs = b.shape[0] // 3
    f = np.zeros((3 * n_projs,))
    s = 0.001 * np.ones((3 * n_projs,))

    q = z[3:7]
    q_norm = np.sqrt(np.sum(q ** 2))
    q = q / q_norm
    R = Quaternion2Rotation(q)
    t = z[:3]

    for j in range(n_projs):
        X = p[3 * j:3 * (j + 1)]
        # Remove measurement error of fixed poses
        b_hat = R @ X + t
        f[3 * j: 3 * (j + 1)] = b_hat / np.sqrt(np.sum(b_hat ** 2))
        if j in hm_ids:
            s[3 * j: 3 * (j + 1)] *= 1000.0

    err = s * (b - f)

    return err


def UpdatePose(z):

    p = z[0:7]
    q = p[3:]

    q = q / np.linalg.norm(q)
    R = Quaternion2Rotation(q)
    t = p[:3]
    P_new = np.hstack([R, t[:, np.newaxis]])

    return P_new


def EPnPRansac(G_p_f, C_b_f_hm, G_P_ctrl, hm_ids, thres=0.01, mode='hm'):
    inlier_thres = thres
    C_T_G_best = None
    inlier_best = np.zeros(G_p_f.shape[1], dtype=bool)
    Nsample=6
    inlier_score_best=0

    for iter in range(500):
        Nfuse = hm_ids.shape[0]
        if mode=='fuse':
            if hm_ids.shape[0] > 0:
                Nfuse = np.random.randint(0, min(Nsample, hm_ids.shape[0]))
                min_set_regvis = np.random.permutation(G_p_f.shape[1]-hm_ids.shape[0])[:(Nsample-Nfuse)]
                min_set_hm = np.random.permutation(hm_ids.shape[0])[:Nfuse]
                min_set = np.concatenate((min_set_hm, min_set_regvis+hm_ids.shape[0]), axis=0)
            else:
                min_set = np.random.permutation(G_p_f.shape[1] - hm_ids.shape[0])[:Nsample]
        elif mode=='hm':
            min_set = np.random.permutation(hm_ids.shape[0])[:Nsample]

        M = construct_control_point_matrix(G_p_f[:, min_set], C_b_f_hm[:, min_set], G_P_ctrl)
        C_R_G_hat, C_t_G_hat = EPnP_solver(M, G_P_ctrl)

        # Evaluate solution
        C_b_f_hat = C_R_G_hat @ G_p_f[:, min_set] + C_t_G_hat
        C_b_f_hat = C_b_f_hat / np.linalg.norm(C_b_f_hat, axis=0)
        if np.sum(np.sum(C_b_f_hat * C_b_f_hm[:, min_set], axis=0) > 1.0-inlier_thres) == min_set.shape[0]:
            # Get inlier
            C_b_f_hat = C_R_G_hat @ G_p_f + C_t_G_hat
            C_b_f_hat = C_b_f_hat / np.linalg.norm(C_b_f_hat, axis=0)
            inlier_mask = np.sum(C_b_f_hat * C_b_f_hm, axis=0) > 1.0 - inlier_thres
            inlier_score = np.sum(inlier_mask[:Nfuse]) * 10 + np.sum(inlier_mask[Nfuse:])
            if inlier_score > inlier_score_best:
                inlier_best = inlier_mask
                C_T_G_best = np.eye(4)
                C_T_G_best[:3, :3] = C_R_G_hat
                C_T_G_best[:3, 3:] = C_t_G_hat
                inlier_score_best = inlier_score

        # if mode == 'hm' and np.sum(inlier_best) > 50:
        #     break

    if np.sum(inlier_best) < 4:
        M = construct_control_point_matrix(G_p_f, C_b_f_hm, G_P_ctrl)
        C_R_G_hat, C_t_G_hat = EPnP_solver(M, G_P_ctrl)
        C_T_G_best = np.eye(4)
        C_T_G_best[:3, :3] = C_R_G_hat
        C_T_G_best[:3, 3:] = C_t_G_hat

    return C_T_G_best, inlier_best


def RunPnPNL(C_T_G, G_p_f, C_b_f, hm_ids, cutoff=0.01):

    z0, b = SetupPnPNL(C_T_G, G_p_f, C_b_f)
    # print('feature: ', track.shape[0])
    # print('nnz: ', S.nnz)
    res = least_squares(
        lambda x: MeasureReprojectionSinglePose(x, b, z0[7:], hm_ids),
        z0[:7],
        verbose=0,
        ftol=1e-4,
        max_nfev=50,
        xtol=1e-8,
        loss='huber',
        f_scale=cutoff
    )
    z = res.x

    # if hm_ids.shape[0] != G_p_f.shape[1] and hm_ids.shape[0] > 0:
    #     # err0 = MeasureReprojectionSinglePose(z0[:7], b, z0[7:], hm_ids=np.empty(1))
    #     err = MeasureReprojectionSinglePose(z, b, z0[7:], hm_ids=np.empty(1))
    #     # print('Reprojection error {} -> {}'.format(np.linalg.norm(err0), np.linalg.norm(err)))
    #     print('Reprojection error hm: ', 1.0 / hm_ids.shape[0] * np.linalg.norm(err[:3*hm_ids.shape[0]]))

    P_new = UpdatePose(z)

    return P_new


def compute_error(C_R_G, C_t_G, C_R_G_hat, C_t_G_hat):

    rot_err = 180 / np.pi * np.arccos(np.clip(0.5 * (np.trace(C_R_G.T @ C_R_G_hat) - 1.0), a_min=-1., a_max=1.))
    trans_err = np.linalg.norm(C_R_G_hat.T @ C_t_G_hat - C_R_G.T @ C_t_G)

    return rot_err, trans_err