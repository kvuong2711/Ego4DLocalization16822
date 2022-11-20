import argparse
import collections.abc as collections
import pickle
from pathlib import Path
from typing import Optional, Union, List
import logging
from utils.parsers import parse_image_lists

logger = logging.getLogger('Ego4DLogger')


def main(output_txt: Path,
         output_dict: Path,
         image_list: Optional[Union[Path, List[str]]] = None,
         ref_list: Optional[Union[Path, List[str]]] = None,):

    if image_list is not None:
        if isinstance(image_list, (str, Path)):
            names_q = parse_image_lists(image_list)
        elif isinstance(image_list, collections.Iterable):
            names_q = list(image_list)
        else:
            raise ValueError(f'Unknown type for image list: {image_list}')
    else:
        raise ValueError('Provide either a list of images or a feature file.')

    self_matching = False
    if ref_list is not None:
        if isinstance(ref_list, (str, Path)):
            names_ref = parse_image_lists(ref_list)
        elif isinstance(image_list, collections.Iterable):
            names_ref = list(ref_list)
        else:
            raise ValueError(
                f'Unknown type for reference image list: {ref_list}')
    else:
        self_matching = True
        names_ref = names_q

    pairs = []
    pairs_dict = {}
    for i, n1 in enumerate(names_q):
        pairs_dict[n1] = []
        for j, n2 in enumerate(names_ref):
            if self_matching and j <= i:
                continue
            pairs.append((n1, n2))
            pairs_dict[n1].append(n2)

    logger.info(f'Found {len(pairs)} pairs.')
    with open(output_txt, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))

    with open(output_dict, 'wb') as f:
        pickle.dump(pairs_dict, f)


