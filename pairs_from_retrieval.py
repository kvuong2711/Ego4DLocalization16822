import logging
from pathlib import Path
import numpy as np
import torch
import argparse
import os
import yaml
from torch.utils.data import DataLoader
from sklearn.neighbors import KDTree
import pickle
from tqdm import tqdm
from utils.dataloader import *


logger = logging.getLogger('Ego4DLogger')


def query_from_db(db_root_folder, visual_db_folder, query_root_folder, query_image_dir, query_descriptor_folder, query_retrieval_filepath_txt, k=5):
    descriptors_cluster_centers = np.load(visual_db_folder / 'data_descriptors_centers.npy')
    descriptors_cluster_centers = torch.tensor(descriptors_cluster_centers, device='cuda')  # KxD
    K, D = descriptors_cluster_centers.shape

    vlad_image_descriptors = np.load(visual_db_folder / 'vlad_image_descriptors.npy')
    image_indices = np.load(visual_db_folder / 'db_image_indices.npy')

    # ######### STEP 2: QUERY IMAGE FROM DATABASE
    kdt = KDTree(vlad_image_descriptors, leaf_size=30, metric='euclidean')

    ego_dataset = GSVDataset(dataset_folder=query_image_dir, resize=[1024], file_ext='.jpg')
    data_loader = DataLoader(dataset=ego_dataset,
                             num_workers=4, batch_size=16,
                             shuffle=False,
                             pin_memory=True)

    VISUALIZATION = True

    query_viz_folder = query_root_folder / 'loc_output' / 'query_viz'
    query_viz_folder.mkdir(parents=True, exist_ok=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # for visualization
    db_images_names = sorted(fnmatch.filter(os.listdir(db_root_folder), '*.jpg'))   # TODO: ext .png

    logger.info('Query image with database images')
    # match_pair = {'db': [], 'query': []}

    pairs = []

    with torch.no_grad():
        for idx, images in enumerate(tqdm(data_loader)):
            images['image'] = images['image'].cuda()

            for j in range(images['image'].shape[0]):
                filename = str(Path(images['name'][j]).stem) + '.npz'

                image_descriptors = np.load(query_descriptor_folder / filename)['descriptors']

                image_descriptors = torch.tensor(image_descriptors).to(device)
                _, N = image_descriptors.shape

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

                # Match and save
                ret_query = kdt.query(vlad_image_descriptor.detach().cpu().numpy().reshape(1, -1), k=k, return_distance=False)[0]

                # visualization
                if VISUALIZATION:
                    all_rows = []
                    for row in range(4):
                        query_imgs = [cv2.resize(cv2.imread(str(query_root_folder / Path(images['name'][j]))), (640, 480)),
                                      np.ones((480, 25, 3)) * 255]
                        for i in range(5 * row, 5 * (row + 1)):
                            retrieved_img = cv2.imread(str(db_root_folder / db_images_names[image_indices[ret_query[i]]]))
                            retrieved_img = cv2.resize(retrieved_img, (640, 480))
                            query_imgs.append(retrieved_img)
                        query_imgs = np.concatenate(query_imgs, axis=1)
                        all_rows.append(query_imgs)
                    all_rows = np.concatenate(all_rows, axis=0)
                    cv2.imwrite(f"{query_viz_folder}/{Path(images['name'][j]).stem}.png", all_rows)

                for qids in range(len(ret_query)):
                    pairs.append((str(Path(images['name'][j]).name), db_images_names[image_indices[ret_query[qids]]]))

    logger.info(f'Found {len(pairs)} pairs.')
    with open(query_retrieval_filepath_txt, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


def main(db_root_folder, visual_db_folder, query_root_folder, query_image_dir, query_descriptor_folder, query_retrieval_filepath_txt, k=5):
    query_from_db(db_root_folder, visual_db_folder, query_root_folder, query_image_dir, query_descriptor_folder, query_retrieval_filepath_txt, k)
