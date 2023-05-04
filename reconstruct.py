#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import time
import torch
import configs.workspace as ws
import data.data_with_labels

import sys
sys.path.insert(0, "./third_party")

from utils.mesh import save_to_ply, colorize_mesh
import utils.mesh
import configs.config_utils
import numpy as np
from networks.dmm_net import DMM



def reconstruct(
    dmm_net,
    teeth_ids,
    num_iterations,
    test_sdf,
    stat,
    num_samples=100000,
    lr=5e-5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_l1 = torch.nn.L1Loss(reduction="sum")
    lat_vecs = dict()

    for label_id in teeth_ids:
        if label_id == 0:
            lat_vecs[label_id] = torch.zeros(1, specs["GumDeformNetworkSpecs"]["latent_dim"]).normal_(mean=0, std=stat).to(device)
        else:
            lat_vecs[label_id] = torch.zeros(1, specs["TeethDeformNetworkSpecs"]["latent_dim"]).normal_(mean=0, std=stat).to(device)
        lat_vecs[label_id].requires_grad = True

    param_lr_lists = list()
    for labelid, lat_vec in lat_vecs.items():
        param_lr_lists.append(
            {
                "params": lat_vec,
                "lr": lr,
            }
        )

    optim = torch.optim.Adam(lr=lr, params=param_lr_lists)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.98)

    loss_num = 0
    dmm_net.eval()

    for e in range(num_iterations):
        if len(test_sdf) > 4 and test_sdf[4] is not None:
            sdf_data, is_on_surface, normal_gt, _ = data.data_with_labels.unpack_sdf_samples_from_ram(
                test_sdf, num_samples
            )
        else:
            sdf_data, is_on_surface, normal_gt = data.data_with_labels.unpack_sdf_samples_from_ram(
                test_sdf, num_samples
            )
        xyz = sdf_data[:, 0:4].to(device)
        is_on_surface = is_on_surface.to(device)
        normal_gt = normal_gt.to(device)

        optim.zero_grad()
        sdf, loss = dmm_net.inference(lat_vecs, xyz, is_on_surface, normal_gt)

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
        loss_num = loss.cpu().data.numpy()

        if e % 50 == 0:
            logging.info("Reconstruction loss: {}".format(loss_num))
            # TODO: Remove the following log
            logging.info("Learning rate: {}".format(optim.param_groups[0]['lr']))
            logging.debug(e)
            logging.debug(lat_vecs[0].norm())

    return loss_num, lat_vecs


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
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=200,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--lr",
        dest="lr",
        default=1e-3,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
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

    #latent_size = specs["CodeLength"]

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    npz_filenames = data.data_with_labels.get_instance_filenames(args.data_source, split)
    logging.info(npz_filenames)

    logging.debug(dmm_net)

    # Set parameters
    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

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

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )

    rgb ={0:[243, 197, 197],
          11: [255, 82, 0],
          12:[255, 31, 90],
          13:[249, 255, 33],
          14:[30, 42, 120],
          15:[255, 23, 0],
          16:[255, 142, 0],
          17:[255, 228, 0],
          18:[6, 255, 0],
          21:[252, 199, 44],
          22:[95, 113, 97],
          23:[0, 161, 171],
          24:[199, 0, 57],
          25:[243, 113, 33],
          26:[192, 226, 24],
          27:[8, 60, 90],
          28:[76, 182, 72]}

    for color_id in rgb.keys():
        rgb[color_id] = torch.Tensor(rgb[color_id]).float() / 255.0

    for ii, npz in enumerate(sorted(npz_filenames)):

        if "npz" not in npz:
            continue

        logging.info("loading {}".format(npz))
        full_filename = os.path.join(args.data_source, npz)

        # load basic SDF data for fitting only
        data_sdf = data.data_with_labels.read_sdf_samples_into_ram(full_filename)

        file_basename = os.path.splitext(npz)[0]
        label_file_name = os.path.join(args.data_source, file_basename + '.txt')
        labels = np.loadtxt(label_file_name).astype(int)
        teeth_ids = np.unique(labels).tolist()

        #labels = data_sdf[0][..., 4].numpy().astype(int)
        #teeth_ids = np.unique(labels).tolist()

        logging.info(teeth_ids)

        min_loss = 1e10
        best_lat_dict = None

        for k in range(repeat):

            if rerun > 1:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, npz[:-4] + "-" + str(k + rerun)
                )
                latent_base_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + "-" + str(k + rerun)
                )
            else:
                mesh_filename = os.path.join(reconstruction_meshes_dir, npz[:-4])
                latent_base_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4]
                )

            if (
                args.skip
                and os.path.isfile(mesh_filename + ".ply")
            ):
                continue

            logging.info("reconstructing {}".format(npz))
            logging.info("repeat no. {}".format(k))

            start = time.time()
            err, latent_vec = reconstruct(
                dmm_net,
                teeth_ids,
                int(args.iterations),
                data_sdf,
                0.0001,
                num_samples=50000,
                lr=float(args.lr)
            )
            logging.debug("Reconstruct time: {}".format(time.time() - start))
            err_sum += err
            logging.debug("Current_error avg: {}".format((err_sum / (ii + 1))))
            logging.debug(ii)

            if min_loss > err:
                min_loss = err
                best_lat_dict = latent_vec

        if not os.path.exists(os.path.dirname(mesh_filename)):
            os.makedirs(os.path.dirname(mesh_filename))

        if not save_latvec_only:
            start = time.time()
            with torch.no_grad():
                mesh_v, mesh_f = utils.mesh.create_mesh(
                    dmm_net, best_lat_dict, mesh_filename, N=512, max_batch=int(2 ** 18)
                )

                colorize_mesh(dmm_net, best_lat_dict, rgb, mesh_v, mesh_f, mesh_filename + "_color.ply")

            logging.info("Total reconstruct time: {} for {}".format(time.time() - start, mesh_filename))

        if not os.path.exists(os.path.dirname(latent_base_filename)):
            os.makedirs(os.path.dirname(latent_base_filename))
        torch.save(best_lat_dict, latent_base_filename + ".pth")
