import logging

import torch
import fnmatch
import argparse
import numpy as np
import os
from sklearn.cluster import KMeans
from tqdm import tqdm


logger = logging.getLogger('Ego4DLogger')


def build_descriptor_center(db_descriptor_folder, visual_db_folder):
    if (visual_db_folder / 'data_descriptors_centers.npy').exists():
        logger.warning('Descriptor center existed, loading from file')
        return

    logger.info('Building descriptor center')
    ############# STEP 0: CONSTRUCT DESCRIPTOR CENTERS
    visual_db_folder.mkdir(parents=True, exist_ok=True)

    MIN_NUM_KEYPOINTS = 300

    total_removal = 0
    all_descriptors = None

    with torch.no_grad():
        # all the image descriptors we got to build the visual database
        matterport_filelist = sorted(fnmatch.filter(os.listdir(db_descriptor_folder), '*.npz'))
        for i, filename in enumerate(tqdm(matterport_filelist)):
            image_descriptors = np.load(db_descriptor_folder / filename)['descriptors']
            _, N = image_descriptors.shape
            if N < MIN_NUM_KEYPOINTS:
                total_removal += 1
                continue
            if all_descriptors is None:
                all_descriptors = np.transpose(image_descriptors, (1, 0))
            else:
                all_descriptors = np.concatenate((all_descriptors,
                                                  np.transpose(image_descriptors, (1, 0))), axis=0)

    logger.info('Compute descriptor centroids with kmeans++ ... ')
    kmeans = KMeans(n_clusters=64, random_state=0, init='k-means++', max_iter=5000, verbose=0).fit(all_descriptors)  # data
    np.save(visual_db_folder / 'data_descriptors_centers.npy', kmeans.cluster_centers_)

    return


def build_vlad_descriptor_tree(db_descriptor_folder, visual_db_folder):
    # make sure the descriptor center from kmeans exists!
    assert (visual_db_folder / 'data_descriptors_centers.npy').exists()
    descriptors_cluster_centers = np.load(visual_db_folder / 'data_descriptors_centers.npy')

    ############# STEP 1: BUILD IMAGE DESCRIPTOR TREE
    logger.info('Building vlad descriptor tree')
    descriptors_cluster_centers = torch.tensor(descriptors_cluster_centers, device='cuda')  # KxD
    K, D = descriptors_cluster_centers.shape

    MIN_NUM_KEYPOINTS = 250

    image_indices = []
    vlad_image_descriptors = []

    logger.info('Compute image descriptor for each database image ... ')
    if (visual_db_folder / 'vlad_image_descriptors.npy').exists() and (visual_db_folder / 'db_image_indices.npy').exists():
        logger.warning('vlad descriptor tree already exists, loading from file!')
    else:
        logger.info('building vlad descriptor tree')
        with torch.no_grad():
            db_filelist = sorted(fnmatch.filter(os.listdir(db_descriptor_folder), '*.npz'))

            for i, filename in enumerate(tqdm(db_filelist)):
                image_descriptors = np.load(db_descriptor_folder / filename)['descriptors']
                image_descriptors = torch.tensor(image_descriptors, device='cuda')
                _, N = image_descriptors.shape
                if N < MIN_NUM_KEYPOINTS:
                    continue

                # compute vlad image descriptor
                assignment_matrix = descriptors_cluster_centers @ image_descriptors  # KxN
                v = torch.max(assignment_matrix, dim=0)
                assignment_mask = assignment_matrix == v.values.reshape(1, N).repeat((K, 1))  # KxN
                assignment_mask = assignment_mask.reshape(1, K, N).repeat((D, 1, 1))
                assignment_mask = assignment_mask.float()

                image_descriptors = image_descriptors.reshape(D, 1, N).repeat((1, K, 1))  # DxKxN
                residual = image_descriptors - torch.transpose(descriptors_cluster_centers, 0, 1).reshape(D, K,
                                                                                                          1)  # DxKxN
                masked_residual = torch.sum(assignment_mask * residual, dim=2)  # DxK

                masked_residual = torch.nn.functional.normalize(masked_residual, p=2, dim=0)
                vlad_image_descriptor = torch.nn.functional.normalize(masked_residual.reshape(1, -1), p=2, dim=1)

                # image_indices.append(int(filename[6:12]))
                image_indices.append(i)
                vlad_image_descriptors.append(vlad_image_descriptor.detach().cpu().numpy())

            vlad_image_descriptors = np.concatenate(vlad_image_descriptors, axis=0)
            image_indices = np.asarray(image_indices)

        np.save(visual_db_folder / 'vlad_image_descriptors.npy', vlad_image_descriptors)
        np.save(visual_db_folder / 'db_image_indices.npy', image_indices)
