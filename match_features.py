import fnmatch
import logging
import os
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import pickle
import torch
import yaml
from tqdm import tqdm
import torch.multiprocessing as mp
from utils.base_model import dynamic_load
import matchers
from utils.parsers import names_to_pair, parse_retrieval

# torch.set_grad_enabled(False)

logger = logging.getLogger('Ego4DLogger')


@torch.no_grad()
def match(pair, matching, opt, output_dir, device):
    name0, name1 = pair[0], pair[1]
    stem0, stem1 = Path(name0).stem, Path(name1).stem
    matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)

    # skip if the matches file exists!
    if matches_path.exists():
        return

    viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt.viz_extension)

    # Handle --cache logic.
    do_viz = opt.viz

    data = {}

    inp0 = np.load(name0)
    for k, v in inp0.items():
        data[k + '0'] = torch.from_numpy(v.__array__()).float().cuda(device=device, non_blocking=True)
    data['image0'] = torch.empty((1,) + tuple(inp0['image_size'])[::-1])

    inp1 = np.load(name1)
    for k, v in inp1.items():
        data[k + '1'] = torch.from_numpy(v.__array__()).float().cuda(device=device, non_blocking=True)
    data['image1'] = torch.empty((1,) + tuple(inp1['image_size'])[::-1])

    data = {k: v[None] for k, v in data.items()}

    pred = matching(data)

    pred['keypoints0'] = data['keypoints0']
    pred['keypoints1'] = data['keypoints1']
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

    # Write the matches to disk.
    out_matches = {'keypoints0': pred['keypoints0'], 'keypoints1': pred['keypoints1'],
                   'matches0': pred['matches0'], 'matching_scores0': pred['matching_scores0']}
    np.savez(str(matches_path), **out_matches)

    # if do_match:
    #     data = {'keypoints0': torch.tensor(np.expand_dims(inp0['keypoints0'], 0)).cuda(non_blocking=True).float(),
    #             'descriptors0': torch.tensor(np.expand_dims(inp0['descriptors0'], 0)).cuda(non_blocking=True).float(),
    #             'image0': torch.tensor(inp0['image0']).cuda(non_blocking=True).float(),
    #             'scores0': torch.tensor(np.expand_dims(inp0['scores0'], 0)).cuda(non_blocking=True).float(),
    #             'keypoints1': torch.tensor(np.expand_dims(inp1['keypoints1'], 0)).cuda(non_blocking=True).float(),
    #             'descriptors1': torch.tensor(np.expand_dims(inp1['descriptors1'], 0)).cuda(non_blocking=True).float(),
    #             'image1': torch.tensor(inp1['image1']).cuda(non_blocking=True).float(),
    #             'scores1': torch.tensor(np.expand_dims(inp1['scores1'], 0)).cuda(non_blocking=True).float()}
    #
    #     print(data.keys())
    #     pred = matching(data)
    #
    #     pred['keypoints0'] = torch.tensor(np.expand_dims(inp0['keypoints0'], 0)).cuda()
    #     pred['keypoints1'] = torch.tensor(np.expand_dims(inp1['keypoints1'], 0)).cuda()
    #     pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    #
    #     kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    #     matches, conf = pred['matches0'], pred['matching_scores0']
    #
    #     # Write the matches to disk.
    #     out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
    #                    'matches': matches, 'match_confidence': conf}
    #     np.savez(str(matches_path), **out_matches)
    #
    # # Keep the matching keypoints.
    # valid = matches > -1
    # mkpts0 = kpts0[valid]
    # mkpts1 = kpts1[matches[valid]]
    # mconf = conf[valid]
    #
    # if do_viz:
    #     # Visualize the matches.
    #     color = cm.jet(mconf)
    #     text = [
    #         'SuperGlue',
    #         'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
    #         'Matches: {}'.format(len(mkpts0)),
    #     ]
    #
    #     # Display extra parameter info.
    #     k_thresh = matching.superpoint.config['keypoint_threshold']
    #     m_thresh = matching.superglue.config['match_threshold']
    #     small_text = [
    #         'Keypoint Threshold: {:.4f}'.format(k_thresh),
    #         'Match Threshold: {:.2f}'.format(m_thresh),
    #         'Image Pair: {}:{}'.format(stem0, stem1),
    #     ]
    #
    #     # TODO: add viz later
    #     # image0_path = Path('/home/tien/Data/MSR/OurScenes/house/seq01/01/image-%06d.color.jpg' % int(name0[-22:-16]))
    #     # image1_path = Path('/home/tien/Data/MSR/OurScenes/house/seq01/01/image-%06d.color.jpg' % int(name1[-22:-16]))
    #
    #     # image0, _, _ = read_image(image0_path, 'cpu', [640, 480], 0, True)
    #     # image1, _, _ = read_image(image1_path, 'cpu', [640, 480], 0, True)
    #     #
    #     # make_matching_plot(
    #     #     image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
    #     #     text, viz_path, opt.show_keypoints,
    #     #     opt.fast_viz, opt.opencv_display, 'Matches', small_text)


def run_process(conf, task, args, output_dir, device):
    # Load the SuperPoint and SuperGlue models.
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)
    model.share_memory()
    for pair in tqdm(task):
        match(pair, matching=model, opt=args, output_dir=output_dir, device=device)


def main(conf, input_pairs, db_descriptor_prefix, query_descriptor_prefix, output_dir, args,
         gpu_list, num_proc_per_gpu):
    logger.info(f'Matching features using GPUs {gpu_list}, {num_proc_per_gpu} procs per GPU')
    assert input_pairs.exists(), input_pairs

    # input_pairs is a dictionary in the form of {q1:[db1, db2, ...], q2:[db1, db3, ...]}
    if input_pairs.suffix == '.txt':
        pairs = parse_retrieval(input_pairs)
        pairs = [(q, r) for q, rs in pairs.items() for r in rs]
        pairs = [(query_descriptor_prefix / q.replace(Path(q).suffix, '.npz'),
                  db_descriptor_prefix / r.replace(Path(r).suffix, '.npz')) for (q, r) in pairs]
    elif input_pairs.suffix == '.pkl' or input_pairs.suffix == '.pickle':
        with open(input_pairs, 'rb') as f:
            d = pickle.load(f)
            pairs = []
            for qname in d:
                for dbname in d[qname]:
                    pairs.append((qname, dbname))
                    pairs = [(query_descriptor_prefix / q.replace(Path(q).suffix, '.npz'),
                              db_descriptor_prefix / r.replace(Path(r).suffix, '.npz')) for (q, r) in pairs]
    else:
        raise ValueError(f'Unknown type (extension) for pairs file: {input_pairs}')

    # Create the output directories if they do not exist already.
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info('Will write matches to directory \"{}\"'.format(output_dir))
    if args.viz:
        logger.info('Will write visualization images to',
              'directory \"{}\"'.format(output_dir))

    mp.set_start_method('spawn', force=True)

    # split task onto multiple gpus
    num_processes = sum(num_proc_per_gpu)
    gpu_indices = []
    for i, gpu_id in enumerate(gpu_list):
        for _ in range(num_proc_per_gpu[i]):
            gpu_indices.append(gpu_id)

    tasks = [pairs[i::num_processes] for i in range(num_processes)]
    processes = []

    for task, gpu_id in zip(tasks, gpu_indices):
        p = mp.Process(target=run_process, args=(conf, task, args, output_dir, gpu_id))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
