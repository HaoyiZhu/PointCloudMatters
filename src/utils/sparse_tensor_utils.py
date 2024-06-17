from collections.abc import Mapping, Sequence

import torch
from torch.utils.data import default_collate


def offset2batch(offset):
    return (
        torch.cat(
            [
                (
                    torch.tensor([i] * (o - offset[i - 1]))
                    if i > 0
                    else torch.tensor([i] * o)
                )
                for i, o in enumerate(offset)
            ],
            dim=0,
        )
        .long()
        .to(offset.device)
    )


def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def point_collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [point_collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: point_collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)


def pcd_collate_fn(batch):
    if "pcds" in batch[0].keys() or (
        "obs" in batch[0].keys() and "pcds" in batch[0]["obs"].keys()
    ):
        if "pcds" in batch[0].keys():
            pcds = [batch[idx].pop("pcds") for idx in range(len(batch))]
        else:
            pcds = [batch[idx]["obs"].pop("pcds") for idx in range(len(batch))]
        return_dict = default_collate(batch)
        pcds = sum(pcds, [])
        pcds = point_collate_fn(pcds)
        if "obs" not in return_dict.keys():
            return_dict["pcds"] = pcds
        else:
            return_dict["obs"]["pcds"] = pcds
    else:
        return_dict = default_collate(batch)
    return return_dict
