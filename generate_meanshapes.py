#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import torch
import configs.workspace as ws

import sys
sys.path.insert(0, "./third_party")
print(sys.path)

import utils.mesh
import configs.config_utils
from networks.dmm_net import DMM


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )

    configs.config_utils.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    configs.config_utils.configure_logging(args)

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    dmm_net = DMM(specs)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir,
            "dmm_" + args.checkpoint + ".pth"
        )
    )
    dmm_net.load_state_dict(saved_model_state["model_state_dict"])
    saved_model_epoch = saved_model_state["epoch"]

    logging.debug(dmm_net)

    label_list = specs["labels"]

    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )
    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    rgb ={0: [243, 197, 197],
          11: [255, 82, 0],
          12: [255, 31, 90],
          13: [249, 255, 33],
          14: [30, 42, 120],
          15: [255, 23, 0],
          16: [255, 142, 0],
          17: [255, 228, 0],
          18: [6, 255, 0],
          21: [252, 199, 44],
          22: [95, 113, 97],
          23: [0, 161, 171],
          24: [199, 0, 57],
          25: [243, 113, 33],
          26: [192, 226, 24],
          27: [8, 60, 90],
          28: [76, 182, 72]}

    for color_id in rgb.keys():
        rgb[color_id] = torch.Tensor(rgb[color_id]).float() / 255.0

    for label_id in label_list:
        mesh_filename = os.path.join(reconstruction_meshes_dir, "meanshape_" + str(label_id).zfill(2))
        with torch.no_grad():
            mesh_v, mesh_f = utils.mesh.create_mesh(
                dmm_net.ref_nets_dict[str(label_id)], None, mesh_filename, N=512, max_batch=int(2 ** 18)
            )

