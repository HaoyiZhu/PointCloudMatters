import os
import warnings
from collections import abc
from importlib import import_module

import numpy as np
import torch


def make_dirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f"custom_imports must be a list but got type {type(imports)}")
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


def interpolate_linear(target_t: int, t1: int, t2: int, x1: np.ndarray, x2: np.ndarray):
    return x1 + (target_t - t1) / (t2 - t1) * (x2 - x1) if t1 != t2 else x1


class TemporalAgg:
    def __init__(
        self,
        apply=False,
        action_dim=8,
        chunk_size=20,
        k=0.01,
    ) -> None:
        self.apply = apply

        if self.apply:
            self.action_dim = action_dim
            self.chunk_size = chunk_size
            self.action_buffer = np.zeros(
                (self.chunk_size, self.chunk_size, self.action_dim)
            )
            self.full_action = False
            self.k = k

    def reset(self):
        self.action_buffer = np.zeros(
            (self.chunk_size, self.chunk_size, self.action_dim)
        )

    def add_action(self, action):
        if not self.full_action:
            t = ((self.action_buffer != 0).sum(1).sum(1) != 0).sum()
            self.action_buffer[t] = action
            if t == self.chunk_size - 1:
                self.full_action = True
        else:
            self.action_buffer = np.roll(self.action_buffer, -1, axis=0)
            self.action_buffer[-1] = action

    def get_action(self):
        actions_populated = (
            ((self.action_buffer != 0).sum(1).sum(1) != 0).sum()
            if not self.full_action
            else self.chunk_size
        )
        exp_weights = np.exp(-np.arange(actions_populated) * self.k)
        exp_weights = exp_weights / exp_weights.sum()
        current_t_actions = self.action_buffer[:actions_populated][
            np.eye(self.chunk_size)[::-1][-actions_populated:].astype(bool)
        ]
        return (current_t_actions * exp_weights[:, None]).sum(0)

    def __call__(self, action):
        if not self.apply:
            return action[0]
        else:
            self.add_action(action)
            return self.get_action()


def build_clip_model(clip_model: str = "ViT-B/16"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load(clip_model, device=device, download_root="./.cache/clip")
    clip_model.requires_grad_(False)
    clip_model.eval()
    return clip_model, device
