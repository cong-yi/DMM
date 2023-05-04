import torch
import torch.nn.functional as F
from utils.math import compute_gradient



def component_sdf_normal_loss(pred_sdf, xyz, is_on_surface, gt_normals, correction, grad_deform, grad_template, label_ids, label_id):

    gradient = compute_gradient(pred_sdf, xyz)

    selection_mask = (is_on_surface != -1) & (label_ids == label_id)
    sdf_constraint = torch.where(selection_mask, pred_sdf, torch.zeros_like(pred_sdf))

    # to avoid flipping
    surface_constraint_mask = (is_on_surface != -1) & (label_ids != label_id) & (pred_sdf < -3e-2)
    sdf_bound_constraint = torch.where(surface_constraint_mask, pred_sdf, torch.zeros_like(pred_sdf))
    sdf_constraint = sdf_constraint + sdf_bound_constraint

    inter_constraint = torch.where(selection_mask, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(selection_mask, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient[..., :1]))
    grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)

    # minimal correction prior
    sdf_correct_constraint = torch.abs(correction)

    # deformation smoothness prior
    grad_deform_constraint = grad_deform.norm(dim=-1)

    # normal consistency prior
    grad_temp_constraint = torch.where(selection_mask,
                                             1 - F.cosine_similarity(grad_template, gt_normals, dim=-1)[ ..., None],
                                             torch.zeros_like(grad_template[..., :1]))

    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,
            'inter': inter_constraint.mean() * 5e2,
            'normal_constraint': normal_constraint.mean() * 1e2,
            'grad_constraint': grad_constraint.mean() * 5e1,
            'sdf_correct_constraint': sdf_correct_constraint.mean() * 5e2,
            'grad_deform_constraint': grad_deform_constraint.mean() * 5,
            'grad_temp_constraint': grad_temp_constraint.mean() * 1e2,
            }


def overall_sdf_normal_loss(pred_sdf, xyz, is_on_surface, gt_normals):

    gradient = compute_gradient(pred_sdf, xyz)

    selection_mask = is_on_surface != -1

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(selection_mask, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(selection_mask, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(selection_mask, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient[..., :1]))
    grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)
    # Exp      # Lapl
    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,
            'inter': inter_constraint.mean() * 2e2,
            'normal_constraint': normal_constraint.mean() * 1e2,
            'grad_constraint': grad_constraint.mean() * 5e1}
