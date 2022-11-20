import argparse
import logging
import sys
from pathlib import Path
import shutil
import multiprocessing
import subprocess
import pprint
from tqdm import tqdm
from utils.read_write_model import read_cameras_binary, read_images_binary
from utils.database import COLMAPDatabase
import numpy as np
from utils.parsers import names_to_pair
from PIL import Image

logger = logging.getLogger('Ego4DLogger')


class CalledProcessError(subprocess.CalledProcessError):
    def __str__(self):
        message = "Command '%s' returned non-zero exit status %d." % (
            ' '.join(self.cmd), self.returncode)
        if self.output is not None:
            message += ' Last outputs:\n%s' % (
                '\n'.join(self.output.decode('utf-8').split('\n')[-10:]))
        return message


# TODO: consider creating a Colmap object that holds the path and verbose flag
def run_command(cmd, verbose=False):
    stdout = None if verbose else subprocess.PIPE
    ret = subprocess.run(cmd, stderr=subprocess.STDOUT, stdout=stdout)
    if not ret.returncode == 0:
        raise CalledProcessError(
            returncode=ret.returncode, cmd=cmd, output=ret.stdout)


def geometric_verification(colmap_path, database_path, pairs_path, verbose):
    logger.info('Performing geometric verification of the matches...')
    cmd = [
        str(colmap_path), 'matches_importer',
        '--database_path', str(database_path),
        '--match_list_path', str(pairs_path),
        '--match_type', 'pairs',
        '--SiftMatching.use_gpu', '1',
        '--SiftMatching.max_num_trials', str(20000),
        '--SiftMatching.min_inlier_ratio', str(0.1)]
    run_command(cmd, verbose)


def create_empty_db(database_path):
    if database_path.exists():
        logging.warning('The database already exists, deleting it.')
        database_path.unlink()
    logging.info('Creating an empty database...')
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()


def import_images(colmap_path, sfm_dir, image_dir, database_path, original_image_ids,
                  single_camera=False, verbose=False):
    logger.info('Importing images into the database...')

    db = COLMAPDatabase.connect(database_path)

    # Scoop through all query images
    images = []
    for g in ['*.jpg', '*.png']:
        images += list(Path(image_dir).glob(g))
    if len(images) == 0:
        raise IOError(f'No images found in {image_dir}.')

    images = sorted(images)
    logger.info(f'Importing images {images}')

    # Check image size consistency (COLMAP will fail if images belonging to same camera have different sizes)
    image_sizes = []
    for image_path in images:
        img = Image.open(image_path)
        if len(image_sizes) == 0:
            image_sizes.append((img.width, img.height))
        else:
            if (img.width, img.height) not in image_sizes:
                raise Exception('Query image sizes mismatch!')

    # We need to create dummy features for COLMAP to import images with EXIF
    dummy_dir = sfm_dir / 'dummy_features'
    dummy_dir.mkdir(parents=True, exist_ok=True)
    for i in images:
        with open(str(dummy_dir / (i.name + '.txt')), 'w') as f:
            f.write('0 128')

    # camera_model = 'SIMPLE_RADIAL_FISHEYE'
    # camera_params = [2320., 2000., 1500., -0.015]
    # camera_model = 'OPENCV_FISHEYE'
    # camera_params = [2320., 2320., 2000., 1500., 0.0, 0.0, 0.0, 0.0]
    # camera_model = 'RADIAL_FISHEYE' # f, cx, cy, k1, k2
    # # camera_params = [2320., 2000., 1500., 0.0, 0.0]
    # camera_params = [2331.45, 1994., 1429., -0.026743, 0.012899] # fifthcraig
    # camera_params = [1000., 960., 540., 0.0, 0.0] # truck cam

    # camera_model = 'RADIAL_FISHEYE'  # f, cx, cy, k1, k2
    # camera_params = [2500., 1980./2., 1980./2., 0.0, 0.0]

    # camera_model = 'SIMPLE_RADIAL'  # f, cx, cy, k
    # camera_params = [2500., 1980./2., 1980./2., 0.0]  # forbescraig
    # camera_model = 'PINHOLE'  # f, cx, cy, k
    # camera_params = [2500., 1980. / 2., 1980. / 2., 0.0]  # forbescraig

    # camera_model = 'SIMPLE_RADIAL'  # f, cx, cy, k
    # camera_params = [2500., 1920./2., 1080./2., 0.0]  # forbescraig

    # camera_model = 'SIMPLE_RADIAL'  # f, cx, cy, k
    # camera_params = [2098.0, 1920./2. - 0.5, 1080./2. - 0.5, 0.0]  # fifth_morewood

    # camera_model = 'RADIAL_FISHEYE'  # f, cx, cy, k1, k2
    # # camera_params = [1200., 1670./2. - 0.5, 938./2. - 0.5, 0.0, 0.0]
    # camera_params = [1800., image_sizes[0][0] // 2 - 0.5, image_sizes[0][1] // 2 - 0.5, 0.0, 0.0]

    camera_model = 'SIMPLE_RADIAL'  # f, cx, cy, k
    camera_params = [1800., image_sizes[0][0] // 2 - 0.5, image_sizes[0][1] // 2 - 0.5, 0.0]  # fifth_morewood

    # camera_model = 'SIMPLE_PINHOLE'  # f, cx, cy, k
    # camera_params = [2098., image_sizes[0][0] // 2 - 0.5, image_sizes[0][1] // 2 - 0.5]  # forbescraig

    cmd = [
        str(colmap_path), 'feature_importer',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--import_path', str(dummy_dir),
        '--ImageReader.single_camera',
        str(int(single_camera)),
        '--ImageReader.camera_model', str(camera_model),
        '--ImageReader.camera_params', ','.join(str(i) for i in camera_params)
    ]

    logging.info('Importing images with command:\n%s', ' '.join(cmd))
    run_command(cmd, verbose)

    # After inserting the new images, there are 2 things we have to do:
    ## 1) Remove keypoints and descriptors from those
    ## 2) Change prior_focal_length to 0

    image_ids = get_image_ids(database_path)
    new_image_ids = list(set(list(image_ids.values())) - set(original_image_ids))
    logger.info(f'New image ids: {new_image_ids}')
    new_camera_ids = []
    for image_id in new_image_ids:
        camera_id = db.execute(f"SELECT camera_id FROM images WHERE image_id={image_id};").fetchall()[0][0]
        new_camera_ids.append(camera_id)

    # Changing prior_focal_length to 0
    # NOTE: only do this for cameras with completely unknown focal length
    for camera_id in new_camera_ids:
        logger.info(f'Setting prior_focal_length of camera_id={camera_id} to 0')
        db.execute(f"UPDATE cameras SET prior_focal_length=0 WHERE camera_id={camera_id}")

    for image_id in new_image_ids:
        db.execute(f"DELETE FROM keypoints WHERE image_id={image_id};")
        db.execute(f"DELETE FROM descriptors WHERE image_id={image_id};")

    db.commit()
    db.close()
    shutil.rmtree(str(dummy_dir))


def import_features(image_ids, database_path, features_path):
    logger.info('Importing features into the database...')
    db = COLMAPDatabase.connect(database_path)

    for image_name, image_id in tqdm(image_ids.items()):
        # logger.info(f'Importing features for {image_name}')
        descriptor_file_path = features_path / image_name.replace(Path(image_name).suffix, '.npz')
        keypoints = np.load(descriptor_file_path)['keypoints'].__array__()
        keypoints += 0.5  # COLMAP origin
        db.add_keypoints(image_id, keypoints)

    db.commit()
    db.close()


def import_matches_query(image_ids, database_path, pairs_path, matches_path,
                         min_match_score=None, skip_geometric_verification=False):
    logger.info('Importing matches into the database...')

    with open(str(pairs_path), 'r') as f:
        pairs = [p.split() for p in f.readlines()]

    db = COLMAPDatabase.connect(database_path)

    matched = set()
    for name0, name1 in tqdm(pairs):
        query_stem, db_stem = Path(name0).stem, Path(name1).stem
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        matches_file_path = matches_path / '{}_{}_matches.npz'.format(query_stem, db_stem)
        if not matches_file_path.exists():
            raise ValueError(f'Could not find matches file {matches_file_path}')
        matches_content = np.load(matches_file_path)
        matches = matches_content['matches0'].__array__()
        valid = matches > -1
        if min_match_score:
            scores = matches_content['matching_scores0'].__array__()
            valid = valid & (scores > min_match_score)
        matches = np.stack([np.where(valid)[0], matches[valid]], -1)

        db.add_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}

        if skip_geometric_verification:
            db.add_two_view_geometry(id0, id1, matches)

    db.commit()
    db.close()


def get_image_ids(database_path):
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images


def resume_reconstruction(colmap_path, sfm_dir, database_path, sparse_input_model_path, image_dir,
                          min_num_matches=None, verbose=False):
    assert sparse_input_model_path.exists()

    latest_model_path = sfm_dir / 'latest_model'
    latest_model_path.mkdir(exist_ok=True, parents=True)

    cmd = [
        str(colmap_path), 'mapper',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--input_path', str(sparse_input_model_path),
        '--output_path', str(latest_model_path),
        '--Mapper.num_threads', str(min(multiprocessing.cpu_count(), 8)),
        '--Mapper.ba_refine_focal_length', 'True',
        '--Mapper.ba_refine_principal_point', 'False',
        '--Mapper.ba_refine_extra_params', 'True',
        '--Mapper.ba_local_num_images', '10',
        '--Mapper.ba_local_max_num_iterations', '40',
        '--Mapper.ba_global_max_num_iterations', '20',
        '--Mapper.abs_pose_min_inlier_ratio', '0.10',
        '--Mapper.abs_pose_min_num_inliers', '10',
        '--Mapper.max_reg_trials', '3',
    ]

    # cmd = [
    #     str(colmap_path), 'mapper',
    #     '--database_path', str(database_path),
    #     '--image_path', str(image_dir),
    #     '--input_path', str(sparse_input_model_path),
    #     '--output_path', str(latest_model_path),
    #     '--Mapper.num_threads', str(min(multiprocessing.cpu_count(), 8)),
    #     '--Mapper.ba_refine_focal_length', 'False',
    #     '--Mapper.ba_refine_principal_point', 'False',
    #     '--Mapper.ba_refine_extra_params', 'False',
    #     '--Mapper.ba_local_num_images', '20',
    #     '--Mapper.ba_local_max_num_iterations', '25',
    #     '--Mapper.ba_global_max_num_iterations', '50',
    #     '--Mapper.abs_pose_min_inlier_ratio', '0.15',
    #     '--Mapper.abs_pose_min_num_inliers', '20',
    #     '--Mapper.max_reg_trials', '3',
    # ]

    if min_num_matches:
        cmd += ['--Mapper.min_num_matches', str(min_num_matches)]
    logging.info('Running the reconstruction with command:\n%s', ' '.join(cmd))
    run_command(cmd, verbose)

    # Largest (latest) model analyzer
    largest_model = latest_model_path
    largest_model_num_cams = len(read_cameras_binary(str(largest_model / 'cameras.bin')))
    largest_model_num_images = len(read_images_binary(str(largest_model / 'images.bin')))
    logging.info(f'Largest model is #{largest_model.name} '
                 f'with {largest_model_num_cams} cameras, '
                 f'{largest_model_num_images} images.')

    stats_raw = subprocess.check_output(
        [str(colmap_path), 'model_analyzer',
         '--path', str(largest_model)])
    stats_raw = stats_raw.decode().split("\n")
    stats = dict()
    for stat in stats_raw:
        if stat.startswith("Registered images"):
            stats['num_reg_images'] = int(stat.split()[-1])
        elif stat.startswith("Points"):
            stats['num_sparse_points'] = int(stat.split()[-1])
        elif stat.startswith("Observations"):
            stats['num_observations'] = int(stat.split()[-1])
        elif stat.startswith("Mean track length"):
            stats['mean_track_length'] = float(stat.split()[-1])
        elif stat.startswith("Mean observations per image"):
            stats['num_observations_per_image'] = float(stat.split()[-1])
        elif stat.startswith("Mean reprojection error"):
            stats['mean_reproj_error'] = float(stat.split()[-1][:-2])

    ## Move the model files outside
    # for filename in ['images.bin', 'cameras.bin', 'points3D.bin']:
    #     shutil.move(str(largest_model / filename), str(sfm_dir / filename))

    return stats


def copy_images_to_one_folder(db_image_dir, query_image_dir, output_image_dir):
    logger.info(f'Copying all query and database images into {output_image_dir}.')
    output_image_dir.mkdir(parents=True, exist_ok=True)
    for db_img_path in db_image_dir.iterdir():
        if db_img_path.is_file() and db_img_path.suffix in ['.png', '.jpg']:
            if not (output_image_dir / db_img_path.name).exists():
                shutil.copy(db_img_path, output_image_dir / db_img_path.name)

    for query_img_path in query_image_dir.iterdir():
        if query_img_path.is_file() and query_img_path.suffix in ['.png', '.jpg']:
            if not (output_image_dir / query_img_path.name).exists():
                shutil.copy(query_img_path, output_image_dir / query_img_path.name)


def main(sfm_dir, db_image_dir, query_image_dir, pairs_file_txt, features_dir, matches_dir,
         colmap_path='colmap', single_camera=False,
         skip_geometric_verification=False,
         min_match_score=None, min_num_matches=None, verbose=False):
    # Initialize the new database
    database = sfm_dir / 'init_database.db'
    shutil.copyfile(database, sfm_dir / 'latest_database.db')
    database = sfm_dir / 'latest_database.db'

    # a few prelim file existence checks
    assert pairs_file_txt.exists(), pairs_file_txt
    assert sfm_dir.exists()
    assert database.exists()

    # get original map image_ids
    original_image_ids_dict = get_image_ids(database)
    original_image_ids_list = list(original_image_ids_dict.values())

    copy_images_to_one_folder(db_image_dir, query_image_dir, sfm_dir / 'images')
    logger.info(f'Images (both query and db) are located at {sfm_dir / "images"}')

    import_images(colmap_path, sfm_dir, query_image_dir, database, original_image_ids_list, single_camera, verbose)

    image_ids = get_image_ids(database)
    new_image_ids_dict = {k: v for k, v in image_ids.items() if k not in original_image_ids_dict}
    logger.info(f'New images dict: {new_image_ids_dict}')

    import_features(new_image_ids_dict, database, features_dir)
    import_matches_query(image_ids, database, pairs_file_txt, matches_dir,
                   min_match_score, skip_geometric_verification)
    if not skip_geometric_verification:
        geometric_verification(colmap_path, database, pairs_file_txt, verbose)

    image_dir = sfm_dir / 'images'
    sparse_input_model_path = sfm_dir / 'sparse'
    stats = resume_reconstruction(
        colmap_path, sfm_dir, database, sparse_input_model_path, image_dir, min_num_matches, verbose)
    if stats is not None:
        stats['num_input_images'] = len(image_ids)
        logging.info('Reconstruction statistics:\n%s', pprint.pformat(stats))
