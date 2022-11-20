import argparse
import os
from streetview_utils import download_panos, query_all_panos, convert_equirect_perspective, circular_grid, visualize_gmap
from tqdm import tqdm
import pathlib
from pathlib import Path
import shutil


def run_download_proc(lat, lng, all_panos_json_path, pruned_panos_json_path, out_pano_folder):
    # Query (search) for all panos
    query_all_panos(lat=lat, lng=lng, all_panos_json_path=all_panos_json_path, pruned_panos_json_path=pruned_panos_json_path, read_only=False)

    # Visualize panos on Google Maps
    visualize_gmap(pruned_panos_json_path, map_title=os.path.basename(os.path.dirname(pruned_panos_json_path)))

    # Download
    download_panos(pano_json=pruned_panos_json_path, out_dir=out_pano_folder)


def run_preprocess_proc(pano_data_dir, perspective_output_data_dir, database_output_data_dir):
    pano_names = sorted(os.listdir(pano_data_dir))
    for pano_name in tqdm(pano_names):
        print('Converting {} to perspective'.format(pano_name))
        convert_equirect_perspective(pano_dir=pano_data_dir, pano_name=pano_name[:-4], output_dir=perspective_output_data_dir)

    # TODO: let's organize a good folder hierarchy for postprocessing
    for filepath in perspective_output_data_dir.glob('**/*.png'):
        newpath = database_output_data_dir / filepath.name
        shutil.copy(filepath, newpath)


def main(args):
    # set up paths
    out_storage = Path(args.raw_storage)
    out_folder = out_storage / args.location_name
    out_folder.mkdir(exist_ok=True, parents=True)
    out_processed_storage = Path(args.processed_storage)
    out_processed_folder = out_processed_storage / args.location_name
    out_processed_folder.mkdir(exist_ok=True, parents=True)

    # run the process
    all_panos_json_path = out_folder / 'all_panos.json'
    pruned_panos_json_path = out_folder / 'pruned_panos.json'
    out_pano_folder = out_folder / 'pano_data'
    out_pano_folder.mkdir(exist_ok=True, parents=True)
    run_download_proc(args.lat, args.lng, all_panos_json_path, pruned_panos_json_path, out_pano_folder)

    # run preprocessing
    out_perspective_folder = out_folder / 'perspective_data'
    out_perspective_folder.mkdir(exist_ok=True, parents=True)
    out_database_img_folder = out_processed_folder / 'database/images'
    out_database_img_folder.mkdir(exist_ok=True, parents=True)
    run_preprocess_proc(pano_data_dir=out_pano_folder,
                        perspective_output_data_dir=out_perspective_folder,
                        database_output_data_dir=out_database_img_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lat', type=float, required=True)
    parser.add_argument('--lng', type=float, required=True)
    parser.add_argument('--location_name', type=str, required=True)
    parser.add_argument('--raw_storage', type=str, required=True)
    parser.add_argument('--processed_storage', type=str, required=True)
    args = parser.parse_args()
    main(args)
