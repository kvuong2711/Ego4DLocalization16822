import numpy as np
from pathlib import Path
from zipfile import ZipFile
from tqdm import tqdm

folder = Path('/mnt/pixel/ilim_data/forbescraig/database/recon_output/matches/superglue')
output_zip = './sampleDir.zip'

with ZipFile(output_zip, 'w') as zipObj:
    for filename in tqdm(list(folder.glob('*.npz'))):
        zipObj.write(filename, filename.name)

