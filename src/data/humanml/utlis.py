
import numpy as np
import torch

from scipy.ndimage import gaussian_filter1d
import torch.nn.functional as F

from src.data.humanml.common.quaternion import qrot, qinv, qbetween
from .scripts.motion_process import recover_from_ric

hml3d_root_joint_l, hml3d_root_joint_r = 0, 4
hml3d_local_joint_pos_l, hml3d_local_joint_pos_r = 4, 67  # 4, 4+21*3
hml3d_joint_rotation_l, hml3d_joint_rotation_r = 67, 193  # 4+21*3, 4+21*3+21*6
hml3d_joint_velocity_l, hml3d_joint_velocity_r = 193, 259  # 4+21*3+21*6+22*3, -4
hml3d_foot_contacts_l, hml3d_foot_contacts_r = 259, 263

kit_root_joint_l, kit_root_joint_r = 0, 4
kit_local_joint_pos_l, kit_local_joint_pos_r = 4, 64  # 4, 4+20*3
kit_joint_rotation_l, kit_joint_rotation_r = 64, 184  # 4+20*3, 4+20*3+20*6
kit_joint_velocity_l, kit_joint_velocity_r = 184, 247  # 4+20*3+20*6+21*3, -4
kit_foot_contacts_l, kit_foot_contacts_r = 247, 251


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_root_6DOF(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel, dtype=data.dtype).to(data)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1) * 2

    r_matrix = euler_to_rotation_matrix(r_rot_ang)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,), dtype=data.dtype).to(data)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    base_r_pos = torch.zeros(data.shape[:-1] + (3,), dtype=data.dtype).to(data)
    base_r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]

    if len(base_r_pos.shape) == 2:
        new_r_pos = torch.einsum("ij,ijk->ik", base_r_pos, r_matrix)
    else:
        new_r_pos = torch.einsum("bij,bijk->bik", base_r_pos, r_matrix)
    new_r_pos = torch.cumsum(new_r_pos, dim=-2)
    new_r_pos[..., 1] = data[..., 3]

    return r_rot_ang, r_matrix, new_r_pos


def euler_to_rotation_matrix(angles):
    Ry = torch.zeros(list(angles.shape) + [3, 3], dtype=angles.dtype).to(angles)
    Ry[..., 0, 0] = torch.cos(angles)
    Ry[..., 2, 2] = torch.cos(angles)
    Ry[..., 0, 2] = torch.sin(angles)
    Ry[..., 2, 0] = -torch.sin(angles)
    Ry[..., 1, 1] = 1
    return Ry


def torch_gaussian_filter1d(input, sigma, kernel_size=None):
    kernel_size = 2 * round(4.0 * sigma) + 1 if kernel_size is None else kernel_size
    # 生成高斯核
    gaussian_kernel = torch.signal.windows.gaussian(kernel_size, std=sigma).to(input)
    gaussian_kernel /= gaussian_kernel.sum()  # 归一化
    # 高斯核需要增加额外的维度以适应 conv1d
    # [out_channels, in_channels, kernel_size]
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size)
    # 扩展高斯核以适配输入通道的数量
    channels = input.shape[1]
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1)
    # 参考filters.gaussian_filter1d(forward, 20, axis=0, mode='nearest')实现的padding
    pad_input = F.pad(input, [kernel_size // 2, kernel_size // 2], "replicate")
    output = F.conv1d(pad_input, gaussian_kernel, groups=channels)
    return output


def recover_pose_from_pmg(raw_data, joint_num):
    r_rot_ang = raw_data[..., 0]
    r_matrix = euler_to_rotation_matrix(r_rot_ang)

    r_pos = raw_data[..., 1:4]
    positions = raw_data[..., 4:(joint_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    if len(positions.shape) == 3:
        positions = torch.einsum("ijk,ikl->ijl", positions, r_matrix)
    else:
        positions = torch.einsum("bijk,bikl->bijl", positions, r_matrix)
    # positions = (positions.unsqueeze(dim=1) @ r_matrix).squeeze(dim=1)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions


def recover_local_velocity_from_joints(positions, face_joint_idx, check=False):
    not_batch = False
    if len(positions.shape) == 3:
        positions = positions.unsqueeze(dim=0)
        not_batch = True
    l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
    across1 = positions[..., r_hip, :] - positions[..., l_hip, :]
    across2 = positions[..., sdr_r, :] - positions[..., sdr_l, :]
    across = across1 + across2
    across = across / torch.sqrt((across ** 2).sum(dim=-1))[..., None]

    forward = torch.cross(torch.tensor([[[0, 1, 0.]]], dtype=positions.dtype, device=positions.device), across, dim=-1)

    if check:
        forward = [gaussian_filter1d(forward[i].numpy(), 20, axis=0, mode='nearest') for i in range(len(forward))]
        forward = torch.from_numpy(np.stack(forward, axis=0)).to(positions)
    else:
        forward = torch_gaussian_filter1d(forward.transpose(-1, -2), 20).transpose(-1, -2)

    forward = forward / torch.sqrt((forward ** 2).sum(dim=-1))[..., None]

    '''Get Root Rotation'''
    target = torch.tensor([[[0, 0, 1.]]], device=positions.device).repeat([1, forward.shape[1], 1]).to(positions)

    root_quat = qbetween(forward, target)
    root_quat[:, 0] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=positions.device)
    local_vel = qrot(root_quat[:, :-1, None].repeat([1, 1, positions.shape[2], 1]),
                     positions[:, 1:] - positions[:, :-1])
    local_vel = local_vel.view(local_vel.shape[0], local_vel.shape[1], -1)
    if not_batch:
        local_vel = local_vel[0]
    return local_vel


def hml3d_to_pmg(raw_data, add_joint_rotation=True, add_foot_concat=True):
    if raw_data.shape[-1] == 263:
        local_joint_pos_l, local_joint_pos_r = hml3d_local_joint_pos_l, hml3d_local_joint_pos_r
        joint_rotation_l, joint_rotation_r = hml3d_joint_rotation_l, hml3d_joint_rotation_r
        foot_contacts_l, foot_contacts_r = hml3d_foot_contacts_l, hml3d_foot_contacts_r
    elif raw_data.shape[-1] == 251:
        local_joint_pos_l, local_joint_pos_r = kit_local_joint_pos_l, kit_local_joint_pos_r
        joint_rotation_l, joint_rotation_r = kit_joint_rotation_l, kit_joint_rotation_r
        foot_contacts_l, foot_contacts_r = kit_foot_contacts_l, kit_foot_contacts_r
    else:
        raise ValueError(f"data shape is not valid!")

    # root joint 6D pose
    r_rot_ang, r_matrix, new_r_pos = recover_root_6DOF(raw_data)
    new_data = torch.cat([r_rot_ang.unsqueeze(dim=-1), new_r_pos, raw_data], dim=-1)
    return new_data


def pmg_to_hml3d(data):
    return data[..., 4:]


def recover_pose_from_pmg(raw_data, joint_num):
    if raw_data.shape[-1] == 267:
        local_joint_pos_l, local_joint_pos_r = hml3d_local_joint_pos_l, hml3d_local_joint_pos_r
    elif raw_data.shape[-1] == 255:
        local_joint_pos_l, local_joint_pos_r = kit_local_joint_pos_l, kit_local_joint_pos_r
    else:
        raise ValueError(f"data shape is not valid!")

    r_rot_ang = raw_data[..., 0]
    r_matrix = euler_to_rotation_matrix(r_rot_ang)

    r_pos = raw_data[..., 1:4]
    positions = raw_data[..., 4+local_joint_pos_l:local_joint_pos_r + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    if len(positions.shape) == 3:
        positions = torch.einsum("ijk,ikl->ijl", positions, r_matrix)
    else:
        positions = torch.einsum("bijk,bikl->bijl", positions, r_matrix)
    # positions = (positions.unsqueeze(dim=1) @ r_matrix).squeeze(dim=1)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions


def convert_motion_representation(data, src_repr, tgt_repr):
    if src_repr == tgt_repr:
        return data
    if src_repr == "hml3d" and tgt_repr == "pmg":
        return hml3d_to_pmg(data)
    if src_repr == "pmg" and tgt_repr == "hml3d":
        return pmg_to_hml3d(data)


def motion_representation_to_joint(data, joint_num, motion_repr, recover_root_from_hml3d=False):
    if motion_repr == "hml3d":
        return recover_from_ric(data, joint_num)
    elif motion_repr == "pmg":
        if recover_root_from_hml3d:
            return recover_from_ric(data[..., 4:], joint_num)
        return recover_pose_from_pmg(data, joint_num)



def scale_data(dataset_name, motion_repr, data, scale):
    if dataset_name != "kit":
        return data
    if motion_repr == "hml3d":
        data[1:4] = data[1:4] * scale
        data[kit_local_joint_pos_l:kit_local_joint_pos_r] *= scale
        data[kit_joint_velocity_l:kit_joint_velocity_r] *= scale
    elif motion_repr == "pmg":
        data[1:4] = data[1:4] * scale
        data[4+1:4+4] = data[4+1:4+4] * scale
        data[4+kit_local_joint_pos_l:4+kit_local_joint_pos_r] *= scale
        data[4+kit_joint_velocity_l:4+kit_joint_velocity_r] *= scale
    return data

@torch.no_grad()
def mask_init_prior(motion, mask, motion_repr):
    if motion.shape[-1] == 263+4:
        joint_rotation_l, joint_rotation_r = hml3d_joint_rotation_l, hml3d_joint_rotation_r
        joint_velocity_l, joint_velocity_r = hml3d_joint_velocity_l, hml3d_joint_velocity_r
    elif motion.shape[-1] == 251+4:
        joint_rotation_l, joint_rotation_r = kit_joint_rotation_l, kit_joint_rotation_r
        joint_velocity_l, joint_velocity_r = kit_joint_velocity_l, kit_joint_velocity_r
    else:
        raise ValueError(f"data shape is not valid!")
    if motion_repr == "pmg":
        tmp = torch.zeros([1, 1, motion.shape[-1]], device=motion.device, dtype=torch.bool)
        # tmp[..., 0] = 1
        tmp[..., 4:4+4] = 1
        tmp[..., joint_rotation_l+4:joint_rotation_r+4] = 1
        tmp[..., joint_velocity_l+4:joint_velocity_r+4] = 1
        tmp = tmp.repeat(motion.shape[0], motion.shape[1], 1)
        tmp[~mask.bool()] = 0
        # tmp = tmp.masked_fill(~mask.unsqueeze(dim=-1).repeat([1, 1, tmp.shape[-1]]), 0)
        return motion.masked_fill(tmp, 0)

    return motion





