import torch
import fnmatch
import argparse
import numpy as np
import os
from sklearn.cluster import KMeans
from tqdm import tqdm
from pathlib import Path

from vdb_builders import vlad_vocabtree


def main(db_descriptor_folder, visual_db_folder):
    vlad_vocabtree.build_descriptor_center(db_descriptor_folder, visual_db_folder)
    vlad_vocabtree.build_vlad_descriptor_tree(db_descriptor_folder, visual_db_folder)
