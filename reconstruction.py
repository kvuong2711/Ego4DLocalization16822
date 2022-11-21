import argparse
import logging
import pickle
from pathlib import Path
import shutil
import multiprocessing
import subprocess
import pprint
import numpy as np
from tqdm import tqdm
from utils.read_write_model import read_cameras_binary
from utils.database import COLMAPDatabase


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


def import_features(image_ids, database_path, features_path):
    logger.info('Importing features into the database...')
    db = COLMAPDatabase.connect(database_path)

    for image_name, image_id in tqdm(image_ids.items()):
        descriptor_file_path = features_path / image_name.replace(Path(image_name).suffix, '.npz')
        keypoints = np.load(descriptor_file_path)['keypoints'].__array__()
        keypoints += 0.5  # COLMAP origin
        db.add_keypoints(image_id, keypoints)

    db.commit()
    db.close()


def import_matches(image_ids, database_path, pairs_path, matches_path,
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
        logger.warning('The database already exists, deleting it.')
        database_path.unlink()
    logger.info('Creating an empty database...')
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()


def import_images(colmap_path, sfm_dir, image_dir, database_path,
                  single_camera=False, verbose=False):
    logger.info('Importing images into the database...')
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f'No images found in {image_dir}.')

    # We need to create dummy features for COLMAP to import images with EXIF
    dummy_dir = sfm_dir / 'dummy_features'
    dummy_dir.mkdir(exist_ok=True)
    for i in images:
        with open(str(dummy_dir / (i.name + '.txt')), 'w') as f:
            f.write('0 128')

    # Load intrinsic and define camera model
    camera_model = 'PINHOLE'
    # intrinsics = np.loadtxt(sfm_dir / 'ba_output/camera_intrinsics.txt')
    # fx, fy, cx, cy = float(intrinsics[0]), float(intrinsics[1]), float(intrinsics[2]), float(intrinsics[3])
    # fx, fy, cx, cy = 960., 960., 960., 540.
    fx, fy, cx, cy = 456.739, 456.646, 479.47, 274.714
    camera_params = [fx, fy, cx, cy]

    cmd = [
        str(colmap_path), 'feature_importer',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--import_path', str(dummy_dir),
        '--ImageReader.single_camera', str(int(single_camera)),
        '--ImageReader.camera_model', str(camera_model),
        '--ImageReader.camera_params', ','.join(str(i) for i in camera_params)]
    logger.info('Importing images with command:\n%s', ' '.join(cmd))

    run_command(cmd, verbose)

    db = COLMAPDatabase.connect(database_path)
    db.execute("DELETE FROM keypoints;")
    db.execute("DELETE FROM descriptors;")
    db.commit()
    db.close()
    shutil.rmtree(str(dummy_dir))


def get_image_ids(database_path):
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images


def run_reconstruction(colmap_path, sfm_dir, database_path, image_dir,
                       min_num_matches=None, verbose=False):
    models_path = sfm_dir / 'models'
    models_path.mkdir(exist_ok=True, parents=True)
    cmd = [
        str(colmap_path), 'mapper',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--output_path', str(models_path),
        '--Mapper.num_threads', str(min(multiprocessing.cpu_count(), 8)),
        '--Mapper.ba_refine_focal_length', 'False',
        '--Mapper.ba_refine_principal_point', 'False',
        '--Mapper.ba_refine_extra_params', 'False',
    ]
    if min_num_matches:
        cmd += ['--Mapper.min_num_matches', str(min_num_matches)]
    logger.info('Running the reconstruction with command:\n%s', ' '.join(cmd))
    run_command(cmd, verbose)

    models = list(models_path.iterdir())
    if len(models) == 0:
        logger.error('Could not reconstruct any model!')
        return None
    logger.info(f'Reconstructed {len(models)} models.')

    largest_model = None
    largest_model_num_images = 0
    for model in models:
        num_images = len(read_cameras_binary(str(model / 'cameras.bin')))
        if num_images > largest_model_num_images:
            largest_model = model
            largest_model_num_images = num_images
    assert largest_model_num_images > 0
    logger.info(f'Largest model is #{largest_model.name} '
                 f'with {largest_model_num_images} images.')

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

    for filename in ['images.bin', 'cameras.bin', 'points3D.bin']:
        shutil.move(str(largest_model / filename), str(sfm_dir / filename))

    sparse_model_path = sfm_dir / 'sparse'
    sparse_model_path.mkdir(exist_ok=True, parents=True)
    for filename in ['images.bin', 'cameras.bin', 'points3D.bin']:
        shutil.copy(str(sfm_dir / filename), str(sparse_model_path / filename))

    return stats


def main(sfm_dir, image_dir, pairs_file_txt, features_dir, matches_dir,
         colmap_path='colmap', single_camera=False, skip_geometric_verification=False,
         min_match_score=None, min_num_matches=None, verbose=False):

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / 'init_database.db'

    create_empty_db(database)
    import_images(colmap_path, sfm_dir, image_dir, database, single_camera, verbose)
    image_ids = get_image_ids(database)
    import_features(image_ids, database, features_dir)
    import_matches(image_ids, database, pairs_file_txt, matches_dir,
                   min_match_score, skip_geometric_verification)
    if not skip_geometric_verification:
        geometric_verification(colmap_path, database, pairs_file_txt, verbose)
    stats = run_reconstruction(
        colmap_path, sfm_dir, database, image_dir, min_num_matches, verbose)
    if stats is not None:
        stats['num_input_images'] = len(image_ids)
        logger.info('Reconstruction statistics:\n%s', pprint.pformat(stats))
