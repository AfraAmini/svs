import os

import torch

MAX_TOL = 10


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def update_noise_variance(eps_p):
    if eps_p > 0.05:
        return eps_p * 0.93  # 0.99
    return 0.05


def early_stopping(last_e, e, tol):
    diff = last_e - e
    diff = torch.sum(diff * diff, dim=-1)
    if diff.mean() < 1e-5:
        tol -= 1
    else:
        tol = MAX_TOL
    if tol == 0:
        return True, tol
    else:
        return False, tol


def pad_mask(lengths, size):
    """
    Create a mask of seq x batch where seq = max(lengths), with 0 in padding locations and 1 otherwise.
    """
    # lengths: bs. Ex: [2, 3, 1]
    max_seqlen = torch.max(lengths)
    expanded_lengths = lengths.unsqueeze(0).repeat(
        (max_seqlen, 1))  # [[2, 3, 1], [2, 3, 1], [2, 3, 1]]
    indices = torch.arange(max_seqlen).unsqueeze(1).repeat((1, lengths.size(0))).to(
        lengths.device)  # [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

    # pad locations are 0. #[[1, 1, 1], [1, 1, 0], [0, 1, 0]]. seqlen x bs
    length_mask = torch.tensor(expanded_lengths > indices).permute(1, 0)
    extended_length_mask = length_mask.unsqueeze(1)
    extended_length_mask2 = extended_length_mask.clone()

    extended_length_mask = extended_length_mask.repeat(1, size, 1)

    extended_length_mask = extended_length_mask.permute(0, 2, 1)
    extended_length_mask2 = extended_length_mask2.permute(0, 2, 1)
    return extended_length_mask, extended_length_mask2


def save_checkpoint(state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)
