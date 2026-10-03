# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
from mind_diffusion import dnnlib


try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        # assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class HoloplaneFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        name_list_file  = None, # List of files to load
        conditions=None, label_scale=1.0, latent_mean=0.0, latent_scale=1.0,
        crop_resolution=0, condition_profile=None,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._conditions = conditions
        self._label_scale = label_scale
        self._latent_mean = latent_mean
        self._latent_scale = latent_scale
        self._crop_resolution = crop_resolution
        self._condition_profile = condition_profile
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        if name_list_file is not None:
            # print("Loading files from list")
            with open(name_list_file, "r") as f:
                self._image_fnames = f.read().splitlines()
            self._image_fnames = [fname.strip() if fname.strip().endswith('.npy') else fname.strip() + '.npy'
                                  for fname in self._image_fnames if fname.strip()]
            missing = set(self._image_fnames) - self._all_fnames
            if missing:
                raise IOError(f'Missing latents: {sorted(missing)[:5]}')
            self._image_fnames = sorted(self._image_fnames)
        else:
            # PIL.Image.init()
            print("Loading all files")
            self._image_fnames = sorted(fname for fname in self._all_fnames if fname.endswith('.npy'))

        if len(self._image_fnames) == 0:
            raise IOError('No numpy files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()


    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None


    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            image = np.load(f, allow_pickle=False)
        if image.ndim == 4 and image.shape[0] == 3:
            image = image.reshape(-1, image.shape[-2], image.shape[-1])
        if image.ndim != 3 or image.shape[-1] != image.shape[-2]:
            raise ValueError(f'{fname}: invalid latent shape {image.shape}')
        if not np.isfinite(image).all():
            raise ValueError(f'{fname}: non-finite latent')
        if self._crop_resolution:
            if self._crop_resolution > image.shape[-1]:
                raise ValueError('crop-resolution exceeds latent size')
            image = image[:, :self._crop_resolution, :self._crop_resolution]
        image = (image.astype(np.float32) - self._latent_mean) * self._latent_scale
        # print(image.shape)
        return image

    def _load_raw_labels(self):
        if self._conditions:
            with open(self._conditions) as f:
                labels = json.load(f)
        else:
            fname = 'dataset.json'
            if fname not in self._all_fnames:
                return None
            with self._open_file(fname) as f:
                labels = json.load(f)
        label_space = 'normalized'
        profile = self._condition_profile
        if isinstance(labels, dict) and 'labels' in labels:
            label_space = labels.get('label_space', 'normalized')
            profile = profile or labels.get('profile')
            if labels.get('components', ['C11', 'C12', 'C44']) != ['C11', 'C12', 'C44']:
                raise ValueError('Condition order must be C11,C12,C44')
            if labels.get('profile') and self._condition_profile and labels['profile'] != self._condition_profile:
                raise ValueError('Condition JSON profile differs from --condition-profile')
            labels = labels['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        if labels.ndim != 2 or labels.shape[1] != 3:
            raise ValueError('Expected [C11,C12,C44] conditions for each sample.')
        if label_space == 'physical':
            from mind_diffusion.conditioning import normalize_C
            labels = normalize_C(labels, profile)
        elif label_space != 'normalized':
            raise ValueError('label_space must be physical or normalized')
        labels = labels.astype(np.float32) * self._label_scale
        if not np.isfinite(labels).all():
            raise ValueError('Non-finite conditions')
        return labels

#----------------------------------------------------------------------------
