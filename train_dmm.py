#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.utils.data as data_utils
import signal
import os
import logging
import json
import time

import sys
sys.path.insert(0, "./third_party")
print(sys.path)

import configs.workspace as ws
from configs.config_utils import add_common_args, configure_logging
import data.data_with_labels as data_with_labels
import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from networks.dmm_net import DMM
from torchmeta.modules import DataParallel

class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))


def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_spec in schedule_specs["InitLr"]:

        if schedule_spec["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_spec["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vecs, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vecs.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def main_function(experiment_directory, continue_from):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("running " + experiment_directory)

    # Initialize the tensorboard log
    now = datetime.datetime.now()
    exp_name = os.path.split(os.path.normpath(experiment_directory))[-1]
    writer = SummaryWriter(os.path.join(experiment_directory, "runs", exp_name + now.strftime("-%Y-%m-%d-%H-%M")))

    # Load the specification file
    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + ''.join(specs["Description"]))

    data_source = specs["DataSource"]

    # output checkpoints
    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )
    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    # learning rates
    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):
        save_model(experiment_directory, "dmm_latest.pth", dmm_net, epoch)
        save_optimizer(experiment_directory, "latest.pth", optim, epoch)
        save_latent_vectors(experiment_directory, "latent_vecs_latest.pth", latent_vecs, epoch)

    def save_checkpoints(epoch):
        save_model(experiment_directory, "dmm_" + str(epoch) + ".pth", dmm_net, epoch)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optim, epoch)
        save_latent_vectors(experiment_directory, "latent_vecs_" + str(epoch) + ".pth", latent_vecs, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    num_epochs = specs["NumEpochs"]

    # create the target folder if not exist
    if not os.path.isdir(os.path.join(experiment_directory, ws.model_params_subdir)):
        os.makedirs(os.path.join(experiment_directory, ws.model_params_subdir))

    # Get file names in the dataset for training
    train_split_file = specs["TrainSplit"]
    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    # Create the training dataset
    sdf_dataset = data_with_labels.SDFSamples(
        data_source, train_split, num_samp_per_scene, load_ram=True
    )

    # Set up specs of the data loader
    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    # Load average tooth centroids
    avg_centers = torch.from_numpy(np.loadtxt(os.path.join(data_source, "avg_centroids.txt"))).to(device)

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    # Compose DMM network
    dmm_net = DMM(specs, avg_centers).to(device)

    # Initialize latent codes for teeth and gums
    latent_vecs = torch.nn.ModuleDict()
    for label_id in dmm_net.label_list:
        if label_id == 0:
            latent_vecs[str(label_id)] = torch.nn.Embedding(num_scenes,
                                                            specs["GumDeformNetworkSpecs"]["latent_dim"]).to(device)
        else:
            latent_vecs[str(label_id)] = torch.nn.Embedding(num_scenes,
                                                            specs["TeethDeformNetworkSpecs"]["latent_dim"]).to(device)
        torch.nn.init.normal_(latent_vecs[str(label_id)].weight, mean=0, std=0.002)
        latent_vecs[str(label_id)] = torch.nn.DataParallel(latent_vecs[str(label_id)])

    # Prepare for the optimizer
    param_lr_lists = list()

    param_lr_lists.append(
        {
            "params": dmm_net.deform_nets_dict.parameters(),
            "lr": lr_schedules[0].initial,
        }
    )

    param_lr_lists.append(
        {
            "params": dmm_net.ref_nets_dict.parameters(),
            "lr": lr_schedules[1].initial,
        }
    )

    param_lr_lists.append(
        {
            "params": latent_vecs.parameters(),
            "lr": lr_schedules[2].initial,
        }
    )

    optim = torch.optim.Adam(param_lr_lists)

    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=lr_schedules[0].interval, gamma=lr_schedules[0].factor)

    start_epoch = 1

    if continue_from is not None:
        logging.info('continuing from "{}"'.format(continue_from))

        saved_model_state = torch.load(
            os.path.join(
                ws.get_model_params_dir(experiment_directory, False),
                "dmm_" + continue_from + ".pth"
            )
        )
        dmm_net.load_state_dict(saved_model_state["model_state_dict"])

        saved_model_epoch = saved_model_state["epoch"]

        saved_latent_state = torch.load(
            os.path.join(
                ws.get_latent_codes_dir(experiment_directory, False),
                "latent_vecs_" + continue_from + ".pth"
            )
        )
        latent_vecs.load_state_dict(saved_latent_state["latent_codes"])

        load_optimizer(
            experiment_directory, continue_from + ".pth", optim
        )

        start_epoch = saved_model_epoch + 1
        for quick_skip in range(start_epoch):
            scheduler.step()

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of DMM parameters: {}".format(
            sum(p.data.nelement() for p in dmm_net.parameters())
        )
    )
    logging.info("lr interval: {}".format(lr_schedules[0].interval))
    total_steps = 0

    dmm_net.train()
    latent_vecs.train()

    for epoch in range(start_epoch, num_epochs + 1):
        start = time.time()
        logging.info("---------------------------------")
        logging.info("epoch {}".format(epoch))
        # TODO: remove it
        logging.info("lr: {}".format(optim.param_groups[0]['lr']))

        log_counter = 0
        for (sdf_data, is_on_surf, normal, centers_tensor), indices in sdf_loader:

            sdf_data = sdf_data.to(device)
            is_on_surf = is_on_surf.to(device)
            normal = normal.to(device)
            centers_tensor = centers_tensor.to(device)
            indices = indices.to(device)

            train_loss, losses_log = dmm_net(latent_vecs, sdf_data, is_on_surf, normal, centers_tensor, indices)

            optim.zero_grad()
            train_loss.backward()
            optim.step()
            if log_counter % 20 == 0:
                logging.info("Total loss: {:10.5f}; Sep loss: {:10.5f}; Embd loss: {:10.5f}; center loss: {:10.5f}; BCE loss: {:10.5f}".format(
                    train_loss.item(), losses_log['sep_loss'], losses_log['embedding_reg_loss'], losses_log['center_loss'], losses_log['blend_loss'])
                )

            writer.add_scalar("total_train_loss", train_loss.item(), total_steps)
            writer.add_scalar("sep_loss", losses_log['sep_loss'], total_steps)
            writer.add_scalar("bce_loss", losses_log['blend_loss'], total_steps)

            log_counter += 1
            total_steps += 1


        scheduler.step()

        end = time.time()

        seconds_elapsed = end - start
        logging.info("epoch training time: {}".format(seconds_elapsed))

        if epoch in checkpoints:
            save_checkpoints(epoch)
            save_latest(epoch)

        writer.add_scalar("learning rate", optim.param_groups[0]['lr'], epoch)

    writer.close()


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train an implicit parametric morphable dental model.")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )

    add_common_args(arg_parser)

    args = arg_parser.parse_args()

    configure_logging(args)

    main_function(args.experiment_directory, args.continue_from)

