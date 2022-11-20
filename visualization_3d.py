import json
import logging
import pickle

import numpy as np

import argparse
import os

import open3d as o3d
from tqdm import tqdm
from utils.read_write_model import read_model, write_model, qvec2rotmat, rotmat2qvec, read_points3D_binary, get_camera_matrix
from localization_colmap import get_image_ids
from utils.database import COLMAPDatabase
from utils.visualization import draw_camera
from utils.geometry import transform_point, transform_pose
from PIL import Image
from matplotlib import pyplot as plt
from utils.LineMesh import LineMesh
from mpl_toolkits.mplot3d import Axes3D
from groundplane_fitting import get_plane_mesh
from scipy.spatial.transform import Rotation as scipy_R

logger = logging.getLogger('Ego4DLogger')


def compute_fov(side_length, f):
    # FOV in radian
    return 2 * np.arctan2(side_length // 2, f)

def rotz(a):
    return np.array([[np.cos(a), -np.sin(a), 0.0],
                     [np.sin(a), np.cos(a), 0.0],
                     [0.0, 0.0, 1.0]])


def get_pcd(points3D, remove_statistical_outlier=True, transform_mat=None):
    # Filter points, use original reproj error here
    max_reproj_error = 3.0
    xyzs = [p3D.xyz for _, p3D in points3D.items() if (
            p3D.error <= max_reproj_error)]
    rgbs = [p3D.rgb for _, p3D in points3D.items() if (
            p3D.error <= max_reproj_error)]

    xyzs = np.array(xyzs)
    rgbs = np.array(rgbs)

    if transform_mat is not None:
        xyzs = transform_point(transform_mat, xyzs.T).T

    # heuristics to crop the point cloud
    median = np.median(xyzs, axis=0)
    std = np.std(xyzs, axis=0)

    num_std = 2
    valid_mask = (xyzs[:, 0] > median[0] - num_std * std[0]) & (xyzs[:, 0] < median[0] + num_std * std[0]) & \
                 (xyzs[:, 1] > median[1] - num_std * std[1]) & (xyzs[:, 1] < median[1] + num_std * std[1]) & \
                 (xyzs[:, 2] > median[2] - num_std * std[2]) & (xyzs[:, 2] < median[2] + num_std * std[2])

    xyzs = xyzs[valid_mask]
    rgbs = rgbs[valid_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzs)
    pcd.colors = o3d.utility.Vector3dVector(rgbs / 255.0)

    # remove obvious outliers
    if remove_statistical_outlier:
        [pcd, _] = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                  std_ratio=2.0)

    return pcd


def plot_points_plane(XYZs, n, ax=None, show=True, plane_color='r'):
    # if ax is None:
    #     fig = plt.figure()
    #     ax = plt.axes(projection='3d')
    #
    # # x = np.linspace(min(XYZs[:, 0]), max(XYZs[:, 0]), 100)
    # # y = np.linspace(min(XYZs[:, 1]), max(XYZs[:, 1]), 100)
    # # X, Y = np.meshgrid(x, y)
    # # Z = (-n[0] * X - n[1] * Y - n[3]) / n[2]
    #
    # x = np.linspace(min(XYZs[:, 0]), max(XYZs[:, 0]), 10)
    # z = np.linspace(min(XYZs[:, 2]), max(XYZs[:, 2]), 10)
    # X, Z = np.meshgrid(x, z)
    # Y = (-n[0] * X - n[2] * Z - n[3]) / n[1]
    #
    # X = X.flatten()
    # Y = Y.flatten()
    # Z = Z.flatten()
    #
    # reduced_mask = (Z >= min(XYZs[:, 2]) - 2) & (Z <= max(XYZs[:, 2]) + 2)
    # X = X[reduced_mask]
    # Y = Y[reduced_mask]
    # Z = Z[reduced_mask]
    #
    # # # ax.plot(XYZs[:, 0], XYZs[:, 1], XYZs[:, 2], 'bo')
    # # if plane_color is not None:
    # #     ax.plot(X, Y, Z, plane_color)
    # # else:
    # #     ax.plot(X, Y, Z)
    # # ax.set_xlabel("X")
    # # ax.set_ylabel("Y")
    # # ax.set_zlabel("Z")
    # # if show:
    # #     plt.show()
    #
    # return X, Y, Z

    if ax is None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

    # x = np.linspace(min(XYZs[:, 0]), max(XYZs[:, 0]), 100)
    # y = np.linspace(min(XYZs[:, 1]), max(XYZs[:, 1]), 100)
    # X, Y = np.meshgrid(x, y)
    # Z = (-n[0] * X - n[1] * Y - n[3]) / n[2]

    x = np.linspace(min(XYZs[:, 0]), max(XYZs[:, 0]), 10)
    z = np.linspace(min(XYZs[:, 2]), max(XYZs[:, 2]), 10)
    X, Z = np.meshgrid(x, z)
    Y = (-n[0] * X - n[2] * Z - n[3]) / n[1]

    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()

    return X, Y, Z


def main(sfm_dir):
    # init_database = sfm_dir / 'init_database.db'
    # latest_database = sfm_dir / 'latest_database.db'

    init_sparse_model_path, input_format = sfm_dir / 'sparse', '.bin'
    init_cameras, init_images, init_points3D = read_model(init_sparse_model_path, ext=input_format)
    logger.info("init_num_cameras: %s", len(init_cameras))
    logger.info("init_num_images: %s", len(init_images))
    logger.info("init_num_points3D: %s", len(init_points3D))

    latest_sparse_model_path, input_format = sfm_dir / 'latest_metric_model', '.bin'
    latest_cameras, latest_images, latest_points3D = read_model(latest_sparse_model_path, ext=input_format)
    logger.info("latest_num_cameras: %s", len(latest_cameras))
    logger.info("latest_num_images: %s", len(latest_images))
    logger.info("latest_num_points3D: %s", len(latest_points3D))

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    dense_scene_mesh_path = latest_sparse_model_path / 'fused.ply'
    logger.info('Dense mesh path: %s', dense_scene_mesh_path)
    if dense_scene_mesh_path.exists():
        logger.info('Using dense mesh')
        dense_scene_mesh = o3d.io.read_point_cloud(os.path.join(latest_sparse_model_path, "fused.ply"))
        [pcd_dense, _] = dense_scene_mesh.remove_statistical_outlier(nb_neighbors=50,
                                                  std_ratio=1.0)
        vis.add_geometry(pcd_dense)  # pcd
    else:
        logger.info('Dense mesh not exists, using sparse')
        pcd = get_pcd(points3D=latest_points3D, remove_statistical_outlier=True, transform_mat=None)
        vis.add_geometry(pcd)  # pcd

    # rectangle_mesh = o3d.io.read_triangle_mesh('/home/kvuong/Downloads/cbox_floor.obj')
    # o3d.visualization.draw_geometries([rectangle_mesh, o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0., 0., 0.])])

    # vis.add_geometry(rectangle_mesh)

    original_image_ids = []
    for img_id, img_info in init_images.items():
        original_image_ids.append(img_id)

    latest_image_ids = []
    for img_id, img_info in latest_images.items():
        latest_image_ids.append(img_id)

    new_image_ids = list(set(latest_image_ids) - set(original_image_ids))

    new_imgid_to_imgname_to_cameraid_dict = {}

    for img_id in new_image_ids:
        new_imgid_to_imgname_to_cameraid_dict[img_id] = {latest_images[img_id].name: latest_images[img_id].camera_id}

    logger.info(f'New image_id - image_name - camera_id dict: \n{new_imgid_to_imgname_to_cameraid_dict}')

    # # Plot plane
    output_plane_equation_txt_file = sfm_dir / 'groundplane_equation.txt'
    refined_plane_eq = np.loadtxt(output_plane_equation_txt_file)
    logging.info('Plane equation: %s', refined_plane_eq)

    X_plane, Y_plane, Z_plane = plot_points_plane(np.asarray(pcd.points), refined_plane_eq)
    XYZ_plane = np.stack((X_plane, Y_plane, Z_plane), axis=1)

    plane_normal = refined_plane_eq[:3].reshape((3, 1))
    e2 = np.array([0, 1, 0], dtype=float).reshape((3, 1))

    g, e = plane_normal, e2
    W_t_P = np.array([0, -refined_plane_eq[-1] / plane_normal[1, 0], 0])  # np.mean(XYZ_plane, axis=0)
    P_R_W = np.eye(3) + 2 * e @ g.T - (1 / (1 + e.T @ g)) * (e + g) @ (e + g).T
    W_R_P = P_R_W.T

    """
    So, TODO: check notes on passive vs. active transformation. This bugged me more than I want to admit!!!
    Here, we use active transformation, transforming the plane from W to P, using W_T_P.
    If we want to change the COORDINATE SYSTEM from W to P (passive), we use P_T_W, but in this active case we use W_T_P.
    """
    # v = np.asarray(rectangle_mesh.vertices)
    # v = v @ W_R_P.T + W_t_P[None, :]
    # rectangle_mesh.vertices = o3d.utility.Vector3dVector(v)
    # vis.add_geometry(rectangle_mesh)

    K = np.array([[500, 0, 320],
                  [0, 500, 240],
                  [0, 0, 1]])
    W_T_P = np.eye(4)
    W_T_P[:3, :3] = W_R_P
    W_T_P[:3, 3] = W_t_P
    print('W_T_P:', W_T_P)

    cam_model = draw_camera(K, W_R_P, W_t_P, K[0, 2] * 2, K[1, 2] * 2, 3.0, [0, 1, 0])
    cam_model = cam_model[0]
    vis.add_geometry(cam_model)

    plane_mesh, coordinate_frame = get_plane_mesh(XYZ_plane, refined_plane_eq[:3])

    vis.add_geometry(coordinate_frame)
    vis.add_geometry(plane_mesh)

    # newly localized cameras
    G_R_Cnew, G_t_Cnew = None, None
    frames = []

    for img_id, imgname_cameraid_dict in new_imgid_to_imgname_to_cameraid_dict.items():
        img_info = latest_images[img_id]
        cam_info = latest_cameras[img_info.camera_id]
        C_R_G, C_t_G = qvec2rotmat(img_info.qvec), img_info.tvec

        # if not 'fifth_craig3' in img_info.name:
        #     continue

        C_T_G = np.eye(4)
        C_T_G[:3, :3], C_T_G[:3, 3] = C_R_G, C_t_G
        # transformed_C_T_G = transform_pose(transform_mat=shift_for_viz_mat, src_mat=C_T_G)
        # C_R_G, C_t_G = transformed_C_T_G[:3, :3], transformed_C_T_G[:3, 3]
        #
        # plane_eq_cam_hack = np.linalg.inv(C_T_G.T) @ refined_plane_eq
        # print('plane_eq_cam_hack:', plane_eq_cam_hack)

        plane_eq_cam_hack = np.linalg.inv(C_T_G.T) @ refined_plane_eq
        gravity = -plane_eq_cam_hack[:3]
        x, y, z = gravity.ravel()
        theta = np.arctan2(z, np.sqrt(
            x ** 2 + y ** 2))  # np.arccos(y / (np.sqrt(x**2 + y**2)))# np.arccos(y)  # np.arctan2(z, np.sqrt(x**2 + y**2))
        phi = np.arctan2(x, y)  # np.sqrt(1 - z**2) * np.arctan2(x, y)
        print('plane_eq_cam_hack:', plane_eq_cam_hack)
        print(theta * 180 / np.pi, phi * 180 / np.pi)

        G_T_C = np.linalg.inv(C_T_G)
        P_T_G = np.linalg.inv(W_T_P)
        P_T_C = P_T_G @ G_T_C
        specified_cam_name = img_info.name
        # print(specified_cam_name, 'P_T_C:', P_T_C)
        P_T_C_flipped = P_T_C.copy()
        P_T_C_flipped[:3, :3] = P_T_C[:3, :3] @ rotz(np.pi)
        print(specified_cam_name, 'P_T_C flipped:', P_T_C_flipped)

        # build intrinsic from params
        cam_params = cam_info.params
        K, dc = get_camera_matrix(camera_params=cam_params, camera_model=cam_info.model)

        print(K, dc)
        print('Horizontal FOV:', np.rad2deg(compute_fov(side_length=1920, f=K[0, 0])))
        print('Vertical FOV:', np.rad2deg(compute_fov(side_length=1080, f=K[0, 0])))

        # invert
        t = -C_R_G.T @ C_t_G
        R = C_R_G.T

        # # rotate z
        # t = rotz(np.pi) @ t
        # R = R @ rotz(np.pi)

        # R[:2, :] *= -1
        # t[:2] *= -1

        G_R_Cnew, G_t_Cnew = R, t

        cam_model = draw_camera(K, R, t, K[0, 2] * 2, K[1, 2] * 2, 0.6, [0, 0.4470, 0.7410])
        frames.extend(cam_model)

    # original cameras (assumed to be PINHOLE)
    for img_id in original_image_ids:
        img_info = latest_images[img_id]
        cam_info = latest_cameras[img_info.camera_id]
        C_R_G, C_t_G = qvec2rotmat(img_info.qvec), img_info.tvec

        # build intrinsic from params
        assert cam_info.model == 'PINHOLE'
        # fx, fy, cx, cy
        K = np.eye(3)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = cam_info.params[0], cam_info.params[1], cam_info.params[2], cam_info.params[3]

        # invert
        t = -C_R_G.T @ C_t_G
        R = C_R_G.T

        cam_model = draw_camera(K, R, t, K[0, 2] * 2, K[1, 2] * 2, 0.1, [0.8500, 0.3250, 0.0980])
        frames.extend(cam_model)

    # # Visualize Traffic4D stuff
    # connections_3d = [[0, 1], [1, 3], [2, 3], [2, 0], [4, 5], [6, 7], [8, 9], [9, 11], [10, 11], [10, 8], [0, 4],
    #                   [4, 8], [1, 5], [5, 9], [2, 6], [6, 10], [3, 7], [7, 11]]
    # connections_3d = np.array(connections_3d)
    # colors_3d = np.zeros((connections_3d.shape[0], 3), dtype=float)
    # for i in range(connections_3d.shape[0]):
    #     colors_3d[i] = np.array([0, 1, 0]) if i < 4 else np.array([1, 0, 0])
    #
    # #### Traffic4D visualization
    # obj_cam_points_pkl_file = '/home/kvuong/ilim-projects/Traffic4D-Release/obj_points_cam_arbinit_optall_allcars.pkl'
    # obj_cam_points = pickle.load(open(obj_cam_points_pkl_file, 'rb'))
    #
    # out_dict = {}
    #
    # # print(obj_cam_points)
    # for i in tqdm(range(obj_cam_points.shape[0])):
    #     obj_global_points = G_R_Cnew @ obj_cam_points[i].T + np.repeat(G_t_Cnew[:, None], repeats=obj_cam_points.shape[1], axis=-1)
    #     obj_global_points = obj_global_points.T
    #     out_dict[i] = obj_global_points.tolist()
    #     # print(obj_global_points.shape)
    #     # print(obj_global_points)
    #
    #     xyzs = np.array(obj_global_points)
    #
    #     line_mesh1 = LineMesh(xyzs,
    #                           connections_3d,
    #                           colors_3d,
    #                           radius=0.05)
    #     line_mesh1_geoms = line_mesh1.cylinder_segments
    #     for linemesh in line_mesh1_geoms:
    #         vis.add_geometry(linemesh)
    #
    # # print(json.dumps(out_dict))
    # with open('/home/kvuong/personal/learn_3js/obj_points_global.json', 'w') as outfile:
    #     json.dump(out_dict, outfile)

    # add geometries to visualizer
    for i in frames:
        vis.add_geometry(i)

    ro = vis.get_render_option()
    # ro.show_coordinate_frame = True
    ro.point_size = 1.0

    ro.light_on = False
    ro.mesh_show_back_face = True
    # ro.mesh_show_wireframe = True

    vis.poll_events()
    vis.update_renderer()
    vis.run()
