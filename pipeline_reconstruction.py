import argparse
from pathlib import Path
from configs.default import get_config, convert_to_dict
import logging
import match_features, pairs_from_retrieval, build_visual_db, extract_features, pairs_from_exhaustive
from utils.misc import merge_queries_pickle_files
import reconstruction
import numpy as np


def setup_logging():
    formatter = logging.Formatter(
        fmt='[%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    logger = logging.getLogger("Ego4DLogger")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False


def setup_paths(cfg):
    paths = {'recon_output': cfg.PATH.RECON_OUTPUT}

    # db paths
    db_recon_output_folder = Path(cfg.PATH.PROCESSED_DB) / cfg.PATH.RECON_OUTPUT
    paths['db_root_folder'] = Path(cfg.PATH.PROCESSED_DB)
    paths['db_color_folder'] = Path(cfg.PATH.PROCESSED_DB) / 'color'  # TODO: hardcoded
    paths['db_descriptor_folder'] = db_recon_output_folder / cfg.PATH.DESCRIPTOR_OUTPUT / cfg.EXTRACTOR.output
    paths['visual_db_folder'] = db_recon_output_folder / cfg.PATH.VDB_OUTPUT / (
            cfg.VDB_BUILDER.output + '_' + cfg.EXTRACTOR.output)

    paths['db_retrieval_filepath_txt'] = db_recon_output_folder / 'db_loc_pairs.txt'
    paths['db_retrieval_filepath_dict'] = db_recon_output_folder / 'db_loc_pairs.pkl'

    paths['db_matches_folder'] = db_recon_output_folder / cfg.PATH.MATCHES_OUTPUT / cfg.MATCHER.output

    paths['sfm_superpoint_superglue'] = db_recon_output_folder / 'sfm_superpoint+superglue'

    return paths


def main(cfg):
    paths = setup_paths(cfg)
    setup_logging()

    # references_images = sorted([str(p.relative_to(paths['db_color_folder'])) for p in (paths['db_color_folder']).glob('*.png')])
    references_images = sorted([str(p.relative_to(paths['db_color_folder'])) for p in (paths['db_color_folder']).glob('*.jpg')])
    print(len(references_images), "mapping images")

    extract_features.main(conf=convert_to_dict(cfg.EXTRACTOR),
                          root_folder=paths['db_color_folder'],
                          output_dir=paths['db_descriptor_folder'],
                          resize=[960],
                          file_ext='.jpg')

    #### Exhaustive ####
    #pairs_from_exhaustive.main(output_txt=paths['db_retrieval_filepath_txt'],
    #                           output_dict=paths['db_retrieval_filepath_dict'],
    #                           image_list=references_images,
    #                           ref_list=None)  # None for self_matching, for reconstruction

    ### Visual DB ####
    build_visual_db.main(db_descriptor_folder=paths['db_descriptor_folder'],
                         visual_db_folder=paths['visual_db_folder'])

    pairs_from_retrieval.main(db_root_folder=paths['db_color_folder'],
                              visual_db_folder=paths['visual_db_folder'],
                              query_root_folder=paths['db_root_folder'],
                              query_image_dir=paths['db_color_folder'],
                              query_descriptor_folder=paths['db_descriptor_folder'],
                              query_retrieval_filepath_txt=paths['db_retrieval_filepath_txt'],
                              k=40)

    match_features.main(conf=convert_to_dict(cfg.MATCHER),
                        input_pairs=paths['db_retrieval_filepath_txt'],
                        db_descriptor_prefix=paths['db_descriptor_folder'],
                        query_descriptor_prefix=paths['db_descriptor_folder'],
                        output_dir=paths['db_matches_folder'],  # query_matches_folder
                        args=argparse.Namespace(starting_index=0, ending_index=-1, viz=False, eval=False, cache=False,
                                                viz_extension='png'),
                        gpu_list=[0],
                        num_proc_per_gpu=[1])

    reconstruction.main(sfm_dir=paths['sfm_superpoint_superglue'],
                        image_dir=paths['db_color_folder'],
                        pairs_file_txt=paths['db_retrieval_filepath_txt'],
                        features_dir=paths['db_descriptor_folder'], matches_dir=paths['db_matches_folder'],
                        colmap_path='colmap',
                        single_camera=True,
                        skip_geometric_verification=False,
                        min_match_score=None, min_num_matches=None,
                        verbose=True)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_config(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='configs/gsv_loc_fc.yml')
    parser.add_argument('opts', help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    main(cfg)
