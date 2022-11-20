import argparse
from pathlib import Path
from configs.default import get_config, convert_to_dict
import logging
import covisibility_query, match_features, build_colmap_bls_model, pairs_from_retrieval, build_visual_db, extract_features, preprocess_data, pairs_from_exhaustive
import groundplane_fitting
import visualization_3d, covisibility_clustering
from utils.misc import merge_queries_pickle_files
import localization_colmap
import geoalign_colmap_lla
import numpy as np

logger = logging.getLogger('Ego4DLogger')


def setup_logging():
    formatter = logging.Formatter(
        fmt='[%(levelname)s - %(asctime)s %(filename)s:%(lineno)d] %(message)s',
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
    paths['db_color_folder'] = Path(cfg.PATH.PROCESSED_DB) / 'images'  # TODO: hardcoded
    paths['db_descriptor_folder'] = db_recon_output_folder / cfg.PATH.DESCRIPTOR_OUTPUT / cfg.EXTRACTOR.output
    paths['visual_db_folder'] = db_recon_output_folder / cfg.PATH.VDB_OUTPUT / (
                cfg.VDB_BUILDER.output + '_' + cfg.EXTRACTOR.output)

    # query paths
    query_loc_output_folder = Path(cfg.PATH.PROCESSED_QUERY) / cfg.PATH.LOC_OUTPUT
    paths['query_root_folder'] = Path(cfg.PATH.PROCESSED_QUERY)
    paths['query_color_folder'] = Path(cfg.PATH.PROCESSED_QUERY) / 'images'  # 'images_static'

    paths['query_descriptor_folder'] = query_loc_output_folder / cfg.PATH.DESCRIPTOR_OUTPUT / cfg.EXTRACTOR.output
    paths['query_retrieval_folder'] = query_loc_output_folder / cfg.PATH.RETRIEVAL_OUTPUT
    paths['query_retrieval_filename'] = cfg.VDB_BUILDER.output + '_retrieved_pairs.pkl'
    paths['query_matches_folder'] = query_loc_output_folder / cfg.PATH.MATCHES_OUTPUT / cfg.MATCHER.output

    paths['query_matches_vlad_folder'] = query_loc_output_folder / cfg.PATH.MATCHES_OUTPUT / (str(cfg.MATCHER.output) + '_vlad')

    paths['query_poses_loc_folder'] = query_loc_output_folder / cfg.PATH.POSES_LOC_OUTPUT

    paths['query_retrieval_filepath_txt'] = query_loc_output_folder / 'loc_pairs.txt'
    paths['query_retrieval_filepath_dict'] = query_loc_output_folder / 'loc_pairs.pkl'

    paths['sfm_superpoint_superglue'] = db_recon_output_folder / 'sfm_superpoint+superglue'

    paths['semantic_segmentation'] = db_recon_output_folder / 'segmentation'
    # paths['sfm_superpoint_superglue'] = db_loc_output_folder / 'sfm_latest'

    return paths


def main(cfg):
    paths = setup_paths(cfg)
    setup_logging()

    # TODO: Design question: Should we move the features for db and query to the same folder?
    #  db_descriptor_folder and query_descriptor_folder to be the same

    # TODO: move this somewhere, probably some utilities functions
    references_images = sorted([str(p.relative_to(paths['db_color_folder'])) for p in (paths['db_color_folder']).glob('*.png')])
    queries_images = sorted([str(p.relative_to(paths['query_color_folder'])) for p in (paths['query_color_folder']).glob('*.jpg')])

    # print('hack the exhaustive a bit')
    # references_images = references_images[:400] + references_images[400::4]

    print(len(references_images), "mapping images")
    print(len(queries_images), "queries images")

    extract_features.main(conf=convert_to_dict(cfg.EXTRACTOR),
                          root_folder=paths['query_color_folder'],
                          output_dir=paths['query_descriptor_folder'],
                          resize=[1024],
                          file_ext='.jpg')

    #### Exhaustive ####
    pairs_from_exhaustive.main(output_txt=paths['query_retrieval_filepath_txt'],
                               output_dict=paths['query_retrieval_filepath_dict'],
                               image_list=queries_images,
                               ref_list=references_images)

    # ### Visual DB ####
    # build_visual_db.main(db_descriptor_folder=paths['db_descriptor_folder'],
    #                      visual_db_folder=paths['visual_db_folder'])
    #
    # pairs_from_retrieval.main(db_root_folder=paths['db_color_folder'],
    #                           visual_db_folder=paths['visual_db_folder'],
    #                           query_root_folder=paths['query_root_folder'],
    #                           query_image_dir=paths['query_color_folder'],
    #                           query_descriptor_folder=paths['query_descriptor_folder'],
    #                           query_retrieval_filepath_txt=paths['query_retrieval_filepath_txt'],
    #                           k=100)

    match_features.main(conf=convert_to_dict(cfg.MATCHER),
                        input_pairs=paths['query_retrieval_filepath_txt'],
                        db_descriptor_prefix=paths['db_descriptor_folder'],
                        query_descriptor_prefix=paths['query_descriptor_folder'],
                        output_dir=paths['query_matches_folder'],  # query_matches_folder
                        args=argparse.Namespace(starting_index=0, ending_index=-1, viz=False, eval=False, cache=False,
                                                viz_extension='png', n_jobs=1),
                        gpu_list=[0,1],
                        num_proc_per_gpu=[1,1])

    localization_colmap.main(
                        sfm_dir=paths['sfm_superpoint_superglue'],
                        db_image_dir=paths['db_color_folder'],
                        query_image_dir=paths['query_color_folder'],
                        pairs_file_txt=paths['query_retrieval_filepath_txt'],
                        features_dir=paths['query_descriptor_folder'],
                        matches_dir=paths['query_matches_folder'],
                        colmap_path='colmap',
                        single_camera=True,
                        skip_geometric_verification=False,
                        min_match_score=None, min_num_matches=None,
                        verbose=True)

    # # visualization_3d.main(sfm_dir=paths['sfm_superpoint_superglue'])
    # groundplane_fitting.main(sfm_dir=paths['sfm_superpoint_superglue'],
    #                          segmentation_folder=paths['semantic_segmentation'])


    # geoalign_colmap_lla.main(colmap_path='colmap',
    #                          db_color_folder=paths['db_color_folder'],
    #                          sfm_dir=paths['sfm_superpoint_superglue'],
    #                          verbose=True)


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
