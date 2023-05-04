#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import pickle

import configs.workspace as ws


def get_all_filenames(data_source):
    npzfiles = []
    for path, dirc, files in os.walk(data_source):
        for name in sorted(files):
            if name.endswith('.npz'):
                npzfiles += [name]
    return npzfiles


def get_instance_filenames(data_source, split):
    npzfiles = []
    for path, dirc, files in os.walk(data_source):
        for name in sorted(files):
            if name in split:
                npzfiles += [name]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    print(mesh_filenames)
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)

    center_dict_name = os.path.splitext(filename)[0] + ".pkl"
    try:
        with open(center_dict_name, 'rb') as f:
            centers_dict = pickle.load(f)
            centers_tensor = torch.zeros(50, 3)
            for key in centers_dict:
                centers_tensor[key, ...] = torch.from_numpy(centers_dict[key])
    except FileNotFoundError:
        logging.info("No center file")
        centers_tensor = None

    surf_tensor = torch.from_numpy(npz["surf"])
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
    normal_tensor = torch.from_numpy(npz["normals"])
    return [
                surf_tensor,
                pos_tensor,
                neg_tensor,
                normal_tensor,
                centers_tensor,
            ]


def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    # surf_tensor = remove_nans(torch.from_numpy(npz["surf"]))
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    surf_tensor = data[0]
    pos_tensor = data[1]
    neg_tensor = data[2]
    normal_tensor = data[3]
    centers_tensor = data[4]

    is_on_surf = torch.zeros((subsample, 1))

    surface_ratio = 0.5

    # split the sample into half
    surf_sampling_num = int(subsample * surface_ratio)

    # random selection
    select_inds = np.random.choice(surf_tensor.shape[0], size=[surf_sampling_num], replace=False)
    sample_surf = surf_tensor[select_inds]
    normal_surf = normal_tensor[select_inds]

    # for Eikonal loss
    is_on_surf[surf_sampling_num:, 0] = -1

    # Samples with positive SDF values
    pos_sampling_num = int((subsample - surf_sampling_num) / 2)
    select_inds = np.random.choice(pos_tensor.shape[0], size=[pos_sampling_num], replace=False)
    sample_pos = pos_tensor[select_inds]

    # Samples with negative SDF values
    neg_sampling_num = subsample - surf_sampling_num - pos_sampling_num
    select_inds = np.random.choice(neg_tensor.shape[0], size=[neg_sampling_num], replace=False)
    sample_neg = neg_tensor[select_inds]

    # Use [-1, -1, -1] as normals for off-the-surface samples
    normal_others = torch.full((pos_sampling_num + neg_sampling_num, 3), -1)
    normals = torch.vstack([normal_surf, normal_others]).float()

    samples = torch.cat([sample_surf, sample_pos, sample_neg], 0)

    if centers_tensor is None:
        return samples, is_on_surf, normals
    else:
        return samples, is_on_surf, normals, centers_tensor


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        load_ram=False
    ):
        self.subsample = subsample

        self.data_source = data_source
        if split is None:
            print("No split file loaded. Use all data.")
            self.npyfiles = sorted(get_all_filenames(data_source))
        else:
            self.npyfiles = get_instance_filenames(data_source, split)

        logging.info(
            "Using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                logging.info(filename)

                self.loaded_data.append(read_sdf_samples_into_ram(filename))


    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx,
            )
        else:
            return unpack_sdf_samples(filename, self.subsample), idx
