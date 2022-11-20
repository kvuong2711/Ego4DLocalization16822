import argparse
import logging
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import pprint
import yaml
from utils.dataloader import AzureKinect, GSVDataset
from utils.geometry import convert_2d_to_3d
from utils.base_model import dynamic_load
import extractors

logger = logging.getLogger('Ego4DLogger')


def main(conf, root_folder, output_dir, resize=[1024], file_ext='.jpg'):
    logger.info(f'Extracting local features with configuration: \n{pprint.pformat(conf)}')

    # Output path
    descriptor_folder = output_dir
    descriptor_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f'Saving local features to: {descriptor_folder}')

    # Setting up dataset
    scan_dataset = GSVDataset(dataset_folder=root_folder, resize=resize, file_ext=file_ext)
    data_loader = DataLoader(dataset=scan_dataset, num_workers=1, batch_size=1, shuffle=False, pin_memory=True)

    # Setting up model (feature extractor)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(extractors, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    with torch.no_grad():
        for idx, images in enumerate(tqdm(data_loader)):
            # forward pass
            images['image'] = images['image'].to(device)
            output = model(images)

            for j in range(images['image'].shape[0]):
                # get the resize scales to properly scale the output from SuperPoint
                resize_scales = images['resize_scales'][j].numpy()

                # result
                name = images['name'][j]

                image_descriptors = output['descriptors'][j].detach().cpu().numpy()  # DxN = 256xNum_KPs
                keypoints = output['keypoints'][j].detach().cpu().numpy()
                scores = output['scores'][j].detach().cpu().numpy()

                # scale the keypoints according to the image size
                keypoints = (keypoints + .5) * resize_scales - .5

                # Write the matches to disk.
                out_matches = {'keypoints': keypoints,
                               'scores': scores,
                               'descriptors': image_descriptors,
                               'keypoint_scale_factor': resize_scales,
                               'image_size': images['original_size'][j].numpy()}
                np.savez(descriptor_folder / f'{Path(name).stem}.npz', **out_matches)

    logger.info('Finished extracting features.')
