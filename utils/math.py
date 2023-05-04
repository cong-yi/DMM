import torch
import numpy as np


def compute_jacobian(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y[..., 0])
    jac = [torch.autograd.grad(y[..., t], x, grad_outputs=grad_outputs, create_graph=True)[0] for t in range(3)]
    jac = torch.stack(jac, dim=-2)
    return jac


def compute_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def compose_affine_trans(s, R, T):
    affine_trans = np.eye(4)
    affine_trans[:3, :3] = s * R
    affine_trans[:3, 3] = T
    return affine_trans


def apply_affine(trans, v):
    return (np.pad(v, [(0, 0), (0, 1)], mode='constant', constant_values=1) @ trans.T)[:, :3]


def screw_axis_to_rt(screw_axis):
    w = screw_axis[..., :3]
    v = screw_axis[..., 3:]
    theta = torch.linalg.norm(w, axis=-1, keepdim=True)#double check
    w = w/theta
    v = v/theta
    W = hat(w)
    # compute R by Rodrigues’ formula
    fac1 = theta.sin()
    fac2 = 1.0 - theta.cos()
    W_square = torch.bmm(W, W)

    R = fac1[..., None] * W + fac2[..., None] * W_square + torch.eye(3, dtype=theta.dtype, device=theta.device)[None]
    T = torch.bmm(theta[..., None] * torch.eye(3, dtype=theta.dtype, device=theta.device)[None] + fac2[..., None] * W + (theta - fac1)[..., None] * W_square, v[..., None])
    return R, T


def hat(v: torch.Tensor) -> torch.Tensor:
    """
    From Pytorch3d:

    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


def get_submesh_by_vertex_labels(mesh, labels, target_id, mask=None):
    v_teeth = mesh.vertices[labels[:] == target_id, :]
    f_teeth = mesh.vertex_faces[labels[:] == target_id]

    f_teeth = set(f_teeth.flatten().tolist())
    if mask is not None:
        f_teeth = f_teeth.intersection(mask)
    f_teeth.discard(-1)

    if len(v_teeth) == 0 or len(f_teeth) == 0:
        print("No submesh labelled as {}".format(target_id))
        return None

    f_teeth = np.array(list(f_teeth))
    submesh = mesh.submesh([f_teeth])[0]

    return submesh