# -*- coding: utf-8 -*-
import torch
from functools import partial


def get_generate_prior_info_fn(prior_info_type: str):
    return partial(generate_prior_info, prior_info_type=prior_info_type)


def generate_prior_info(motion: torch.Tensor, motion_length: torch.Tensor=None, prior_info_type: str='1_first'):
    N, L = motion.shape[:2]
    device = motion.device
    motion_mask = torch.zeros([N, L], dtype=torch.long, device=device)

    if motion_length is None:
        motion_length = torch.zeros([N,], dtype=torch.long, device=device)
        motion_length = motion_length.fill_(L)
    else:
        motion_length = motion_length.clone().detach()

    if prior_info_type == '0':
        return motion_mask
    elif prior_info_type == '1_first':
        ids = torch.zeros([N, 1], dtype=torch.long, device=device)
    elif prior_info_type == '1_end':
        ids = motion_length[:, None] - 1
    elif prior_info_type == '1_mid':
        ids = (motion_length // 2)[:, None]
    elif prior_info_type == '1_random':
        random_num = torch.rand(N, device=device)
        ids = torch.floor(random_num * motion_length).long()
        ids = ids.unsqueeze(dim=1)
    elif prior_info_type == '2_border':
        ids = torch.cat([torch.zeros([N, 1], dtype=torch.long, device=device),
                         motion_length[:, None] - 1,
                         ], dim=-1)
    elif prior_info_type == '2_random':
        ids = get_random_number(motion_length, 2)
    elif prior_info_type == '4_uniform':
        step = (motion_length // 4)[:, None]
        ids = torch.cat([torch.zeros([N, 1], dtype=torch.long, device=device),
                         step, step*2, motion_length[:, None] - 1,
                         ], dim=-1)
    elif prior_info_type == '4_random':
        ids = get_random_number(motion_length, 4)
    else:
        print('given info type error!')
        exit()
    motion_mask = motion_mask.scatter(-1, ids, 1)
    # motion_mask = motion_mask[:, None, :].repeat([1, C, 1])
    return motion_mask


def generate_predict_map(prior_indicate: torch.Tensor, motion_length: torch.Tensor,
                         num_prior=None, k_max=0, k_cur=0, skip_short=True, predict_keyframe=False):
    B, L = prior_indicate.shape
    device = prior_indicate.device
    if num_prior == 0:
        assert k_max == 1
        predict_map = torch.zeros([B, L], dtype=torch.long).to(device) + 2
        mask = torch.arange(L, device=device).repeat([B, 1]) < motion_length[..., None]
        predict_map[mask] = 1
        return predict_map
    tmp = torch.arange(1, L+1).repeat([B, 1]).to(device)
    if num_prior is None:
        num_prior = torch.sum(prior_indicate, dim=-1)
        ids, _ = torch.topk(prior_indicate[:, :] * tmp, num_prior.max())
        ids[ids == 0] = 1000
    else:
        ids, _ = torch.topk(prior_indicate[:, :]*tmp, num_prior)
    dis, _ = torch.abs(tmp[..., None] - ids[:, None, :]).min(dim=-1)
    out_of_length_mask = tmp > motion_length[..., None]
    dis[out_of_length_mask] = -1

    if k_max == 0:
        return dis

    max_dis, _ = dis.max(dim=-1)
    step_size = torch.ceil(max_dis / k_max)
    dis = dis / step_size[..., None]
    dis[dis < 0] = -1
    dis = torch.ceil(dis)
    if k_cur == 0:
        random_k = torch.randint(1, k_max+1, [B, 1]).to(device)
    else:
        random_k = k_cur
    predict_map = torch.zeros_like(dis).to(device) + 2
    predict_map[dis < random_k] = 0
    predict_map[dis == random_k] = 1
    if skip_short:
        predict_map[motion_length < 16] = 1
    if k_cur == 1:
        predict_map[dis < 1] = 0
    if predict_keyframe:
        # only predict keyframe when k = 1
        index = (prior_indicate * (random_k == 1)).bool()
        predict_map[index] = 1
    predict_map[dis < 0] = 3
    return predict_map.long() # [N, L]

# get random and no replace number
def get_random_number(a, n):
    random_numbers = []
    for x in a:
        if x < n:
            random_numbers.append(torch.zeros(n, dtype=torch.long))
        else:
            random_numbers.append(torch.randperm(x, dtype=torch.long)[:n])
    ret = torch.stack(random_numbers, dim=0).to(a.device)
    return ret


if __name__ == '__main__':
    motion = torch.rand([4, 22, 8])
    motion_length = torch.tensor([12, 14, 20, 16], dtype=torch.long)
    # prior_indicate = generate_prior_info(motion, motion_length, '2_random')
    prior_indicate = generate_prior_info(motion, motion_length, '1_random')
    print(prior_indicate)
    print("-"*50)
    for k_cur in range(1, 4):
        predict_map = generate_predict_map(prior_indicate, motion_length, 1, k_max=3, k_cur=0, predict_keyframe=True)
        print(predict_map)
        # print(predict_map.shape)
        predict_map = generate_predict_map(prior_indicate, motion_length, 1, k_max=3, k_cur=k_cur,
                                           predict_keyframe=False)
        print(predict_map)
        # print(predict_map.shape)
        print("#"*50)
        # break