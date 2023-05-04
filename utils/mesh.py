#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch
from utils.decode import decode_sdf
from utils.math import screw_axis_to_rt


def create_mesh(
    decoder, latent_vec, filename, N=256, max_batch=32 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (
            decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    return convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file
    save_to_ply(mesh_points, None, faces, ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
    return mesh_points, faces


def colorize_mesh(dmm_net, lat_vec_dict, rgb, mesh_v, mesh_f, filename):
    mesh_v = torch.tensor(mesh_v).cuda()
    batch_xyz = mesh_v.float().chunk(100)
    mesh_v_color = list()
    for xyz in batch_xyz:
        blend_weight_list = list()
        color_list = list()
        for label_id in lat_vec_dict.keys():
            embeddings = lat_vec_dict[label_id]
            v_color = rgb[label_id].repeat(xyz.shape[0], 1)
            color_list.append(v_color)

            deform_model_output = dmm_net.deform_nets_dict[str(label_id)](xyz, embeddings)
            blend_weights = deform_model_output[..., -1]
            blend_weight_list.append(blend_weights)

        predict_weights = torch.stack(blend_weight_list, dim=-1)
        predict_weights = torch.sigmoid(predict_weights).squeeze()

        predict_weights = torch.nn.functional.normalize(predict_weights, dim=-1, p=1)

        color_blend = torch.stack(color_list, dim=-1).cuda()
        color_blend = torch.sum(color_blend * predict_weights[..., None, :], dim=-1).squeeze()
        mesh_v_color.append(color_blend)
    mesh_v_color = (torch.concat(mesh_v_color, dim=0).cpu().numpy() * 255).astype(np.uint8)
    save_to_ply(mesh_v.cpu().numpy(), mesh_v_color, mesh_f, filename)


def save_to_ply(verts, verts_color, faces, ply_filename_out):
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    if verts_color is None:
        verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

        for i in range(0, num_verts):
            verts_tuple[i] = tuple(verts[i, :])
    else:
        verts_tuple = np.zeros(
            (num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])

        for i in range(0, num_verts):
            verts_tuple[i] = (verts[i][0], verts[i][1], verts[i][2],
                              verts_color[i][0], verts_color[i][1], verts_color[i][2])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

