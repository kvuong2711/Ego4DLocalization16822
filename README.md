## Egocentric video relocalization pipeline

Reconstruction pipeline:

1) Download data from https://www.dropbox.com/s/g9q30t4nhwkmb1e/temp_data.zip?dl=0

2) Change path in `configs/gsv_loc_shaler.yml`: `PROCESSED_DB` and `PROCESSED_QUERY` to the extracted data folder

3) Run `python pipeline_reconstruction.py --config_file configs/gsv_loc_shaler.yml`

Please refer to https://github.com/cvg/Hierarchical-Localization for setting up conda environment.