import cv2
import numpy as np
import sys
import os
import glob
import json
import shutil
import argparse
from PIL import Image
from joblib import Parallel, delayed


def undistort(filename, downsample_factor, camera_matrix, distortion_params, output_path):
    I = cv2.imread(filename)
    if downsample_factor != 1:
        new_size = (I.shape[1] // downsample_factor, I.shape[0] // downsample_factor)
        I = cv2.resize(I, dsize=new_size, interpolation=cv2.INTER_CUBIC)

    # undistort operation
    I2 = cv2.undistort(I, camera_matrix, distortion_params)

    base_name = os.path.splitext(os.path.basename(filename))[0]
    new_color_name = os.path.join(output_path, base_name + '.jpg')
    cv2.imwrite(new_color_name, I2)


def undistort_serial(filenames, downsample_factor, camera_matrix, distortion_params, output_path):
    for filename in filenames:
        undistort(filename, downsample_factor, camera_matrix, distortion_params, output_path)


def generate_undistorted_color_images(dataset_path, intrinsics, downsample_factor=1):
    if intrinsics['type'] != 'K4A_CALIBRATION_LENS_DISTORTION_MODEL_BROWN_CONRADY':
        raise NotImplementedError

    k1 = intrinsics['k1']
    k2 = intrinsics['k2']
    k3 = intrinsics['k3']
    k4 = intrinsics['k4']
    k5 = intrinsics['k5']
    k6 = intrinsics['k6']
    p1 = intrinsics['p1']
    p2 = intrinsics['p2']

    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    if downsample_factor != 1:
        fx = fx / downsample_factor
        fy = fy / downsample_factor
        cx = cx / downsample_factor
        cy = cy / downsample_factor
        print('Effective fc = [{0}, {1}], cc=[{2}, {3}].'.format(fx, fy, cx, cy))

    camera_matrix = np.eye(3)
    camera_matrix[0, 0] = fx
    camera_matrix[1, 1] = fy
    camera_matrix[0, 2] = cx
    camera_matrix[1, 2] = cy

    distortion_params = np.array([k1, k2, p1, p2, k3, k4, k5, k6])

    output_path = os.path.join(dataset_path, 'color_undistorted')
    os.makedirs(output_path, exist_ok=True)
    images_path = os.path.join(dataset_path, 'color')

    # Copy the timestamps file.
    shutil.copyfile(os.path.join(images_path, 'timestamps.txt'),
                    os.path.join(output_path, 'timestamps.txt'))

    n_proc = 8
    parallel_executor = Parallel(n_jobs=n_proc, verbose=1)

    tasks = []
    filenames = glob.glob(os.path.join(images_path, '*.jpg'))
    n_tasks = len(filenames) // 5
    batch_filenames = [filenames[i::n_tasks] for i in range(n_tasks)]

    for filenames in batch_filenames:
        tasks.append(delayed(undistort_serial)(filenames, downsample_factor, camera_matrix, distortion_params, output_path))

    parallel_executor(tasks)


# Coverts list of strings to list of list[filename, timestamp]
def convert_lines_to_list(lines):
    lst = []
    for line in lines:
        splt = line.split(' ')
        filename = splt[0]
        ts_str = splt[1]
        lst.append([filename.strip(), float(ts_str.strip())])
    return lst


# Converts filename and timestamps to list of timestamps (numpy.array) and a
# mapping from the timestamps to the filename.
def convert_list_to_map(list_of_files):
    all_timestamps = [x[1] for x in list_of_files]
    return np.array(all_timestamps)


def find_closest_times_to_images(color_times, depth_times):
    resulting_indices = []
    time_differences = []
    for time in color_times:
        smallest_index = np.argmin(np.abs(depth_times - time))
        if abs(time - depth_times[smallest_index]) > 0.5:
            smallest_index = -1

        time_differences.append(abs(time - depth_times[smallest_index]))
        resulting_indices.append(smallest_index)

    time_differences = np.array(time_differences)
    print('Average time difference = {}, max time difference = {}'.format(np.mean(time_differences), np.max(time_differences)))
    return resulting_indices


def find_correspondences(dataset_path):
    images_path = os.path.join(dataset_path, 'color')
    depths_path = os.path.join(dataset_path, 'depth')

    # Get the timestamp files
    image_timestamps_filename = os.path.join(images_path, 'timestamps.txt')
    depth_timestamps_filename = os.path.join(depths_path, 'timestamps.txt')

    with open(image_timestamps_filename, 'r') as f:
        all_image_lines = f.readlines()

    print(depth_timestamps_filename)
    with open(depth_timestamps_filename, 'r') as f:
        all_depth_lines = f.readlines()

    print(len(all_image_lines), len(all_depth_lines))
    color_list = convert_lines_to_list(all_image_lines)
    depth_list = convert_lines_to_list(all_depth_lines)

    color_times = convert_list_to_map(color_list)
    depth_times = convert_list_to_map(depth_list)

    closest_depth_indices = find_closest_times_to_images(color_times, depth_times)

    correspondence_list = []
    for color_index in range(0, len(color_list)):
        depth_index = closest_depth_indices[color_index]
        if depth_index == -1:
            continue

        color_image_name = color_list[color_index][0]
        depth_image_name = depth_list[depth_index][0]
        correspondence_list.append([color_image_name, depth_image_name])

    with open(os.path.join(dataset_path, 'color_to_depth_correspondences.txt'), 'w') as f:
        for item in correspondence_list:
            f.write('{0}, {1}\n'.format(item[0], item[1]))


def process_dataset(dataset_path, rgb_intrinsics_path, downsample_rgb_factor):
    if rgb_intrinsics_path == '':
        rgb_intrinsics_path = os.path.join(dataset_path, 'rgb_intrinsics.json')

    # First read the camera intrinsics.
    with open(rgb_intrinsics_path, 'r') as fp:
        rgb_intrinsics = json.load(fp)

    print('Finding correspondences...')
    print(dataset_path)
    find_correspondences(dataset_path)
    print('Done')
    # Now generate undistorted color images.
    print('Generating undistorted color images...')
    generate_undistorted_color_images(dataset_path, rgb_intrinsics, downsample_rgb_factor)
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a dataset collected via MARS Kinect Azure dataset collection tool.')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the dataset.')
    parser.add_argument('--rgb_intrinsics', type=str, default='',
                        help='Path to the rgb_intrinsics.json file.')
    parser.add_argument('--downsample_rgb_factor', type=int, default=2,
                        help='Downsample the undistorted RGB image by this factor.')

    args = parser.parse_args()

    process_dataset(args.dataset_path, args.rgb_intrinsics, args.downsample_rgb_factor)
