import os
import math
import random
import hashlib
import skimage.io
import skimage.measure
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


def load_npy_file(npy_path):
    npy = np.load(npy_path)
    t1w, t2w, pd, t1, t2 = np.transpose(npy, (2, 0, 1))
    return t1w, t2w, pd, t1, t2



class T1T2Dataset(Dataset):
    def __init__(self, cfg, train_or_test, transforms, fold=1):
        self.cfg = cfg
        self.train_or_test = train_or_test
        self.transforms = transforms
        self.fold = fold

        self.n_folds = cfg['training']['n_folds']
        self.mixed_precision = cfg['training'].get('mixed_precision', False)

        self.data_dir = cfg['data']['npz_path_trainval']

        self.dates = self.load_dates()
        self.sequences = self.load_sequences()

    def load_dates(self):
        """Get each unique date in the PNG directory and split into train/test using seeding for reproducibility"""

        def get_train_test_for_date(date):
            randnum = int(hashlib.md5(str.encode(date)).hexdigest(), 16) / 16 ** 32
            test_fold = math.floor(randnum * self.n_folds) + 1
            if test_fold == self.fold:
                return 'test'
            else:
                return 'train'

        assert self.train_or_test in ('train', 'test')
        images = sorted(glob(os.path.join(self.data_dir, f"**/*__combined.npz"), recursive=True))
        dates = list({os.path.basename(i).split('__')[0] for i in images})
        dates = [d for d in dates if get_train_test_for_date(d) == self.train_or_test]
        return dates

    def load_sequences(self):
        """Get a list of tuples of (imgpath, labpath)"""
        sequences = []
        for date in sorted(self.dates):
            imgpaths = sorted(glob(os.path.join(self.data_dir, f"**/{date}__*__combined.npz"), recursive=True))  # Get all images
            sequences.extend(imgpaths)
        print(f"{self.train_or_test.upper():<5} FOLD {self.fold}: Loaded {len(sequences)} over {len(self.dates)} dates")
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        n_channels_keep_img = len(self.cfg['export']['source_channels'])  # May have exported more channels to make PNG

        imgpath = self.sequences[idx]
        img = np.load(imgpath)['dicom']
        lab = np.load(imgpath)['label']

        imglab = np.dstack((img, lab))

        trans = self.transforms(image=imglab)['image']

        imglab = trans.transpose([2, 0, 1])
        img = imglab[:n_channels_keep_img]
        lab = imglab[n_channels_keep_img:]

        # BELOW CURRENTLY NOT NEEDED AS WE ARE NOT NORMALISING SO LABELS SHOULD STILL BE VALID
        # Scale between 0 and 1, as normalisation will have denormalised, and possibly some augs too, e.g. brightness
        # lab = (lab - lab.min())
        # lab = lab / (lab.max() + 1e-8)

        x = torch.from_numpy(img).float()
        y = torch.from_numpy(lab).float()

        if self.mixed_precision:
            x = x.half()
            y = y.half()
        else:
            x = x.float()
            y = y.float()

        return x, y, imgpath

    def get_numpy_paths_for_sequence(self, sequence_tuple):
        npy_root = self.cfg['export']['npydir']
        imgpath = sequence_tuple
        datefolder, studyfolder, npyname, _ext = os.path.basename(imgpath).split('__')
        return os.path.join(npy_root, datefolder, studyfolder, npyname + '.npy')
