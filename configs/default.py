from typing import List, Optional, Union

import yacs.config


# Default config node, copied from Facebook's Habitat
class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


CN = Config

CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
# -----------------------------------------------------------------------------
# DATASET PATHS (SUPPOSED TO BE OVERWRITTEN BY USERS)
# -----------------------------------------------------------------------------
_C.PATH = CN()
_C.PATH.RAW_DB = 'raw_db_path'
_C.PATH.PROCESSED_DB = 'processed_db_path'
_C.PATH.RAW_QUERY = 'raw_query_path'
_C.PATH.PROCESSED_QUERY = 'processed_query_path'

# -----------------------------------------------------------------------------
# THESE PATHS ARE RELATIVE TO THE PATHS ABOVE AND NOT SUPPOSED TO BE CHANGED
# -----------------------------------------------------------------------------
_C.PATH.LOC_OUTPUT = 'loc_output'
_C.PATH.DESCRIPTOR_OUTPUT = 'descriptor'
_C.PATH.MATCHES_OUTPUT = 'matches'
_C.PATH.RETRIEVAL_OUTPUT = 'retrieval'
_C.PATH.VDB_OUTPUT = 'visual_database'
_C.PATH.POSES_LOC_OUTPUT = 'poses_loc'

# Reconstruction stuff
_C.PATH.RECON_OUTPUT = 'recon_output'

# -----------------------------------------------------------------------------
# THESE PATHS ARE RELATIVE TO THE PATHS ABOVE AND NOT SUPPOSED TO BE CHANGED
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# PREPROCESSING CONFIGURATION
# -----------------------------------------------------------------------------
_C.PREPROCESS = CN()
_C.PREPROCESS.COPY_DATA = True
_C.PREPROCESS.DEPTH_FUSION = True
_C.PREPROCESS.SUBSAMPLE_RAWDATA_ONLY = False


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    return config


_VALID_TYPES = {tuple, list, str, int, float, bool}


def convert_to_dict(cfg_node, key_list=[]):
    """ Convert a config node to dictionary """
    if not isinstance(cfg_node, yacs.config.CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print("Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES), )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict