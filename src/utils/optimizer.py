import collections.abc
import math
import re
from collections import defaultdict
from copy import deepcopy
from itertools import chain, islice
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.utils import RankedLogger
from src.utils.registry import Registry

OPTIMIZERS = Registry("optimizers")


OPTIMIZERS.register_module(module=torch.optim.SGD, name="SGD")
OPTIMIZERS.register_module(module=torch.optim.Adam, name="Adam")
OPTIMIZERS.register_module(module=torch.optim.AdamW, name="AdamW")

_logger = RankedLogger(__name__, rank_zero_only=True)

# optimizers to default to multi-tensor
_DEFAULT_FOREACH = {
    "lion",
}

MATCH_PREV_GROUP = (99999,)


def build_optimizer(cfg, model, param_dicts=None):
    cfg = deepcopy(cfg)

    if param_dicts is None:
        cfg.params = model.parameters()
    else:
        cfg.params = [dict(names=[], params=[], lr=cfg.lr)]
        for i in range(len(param_dicts)):
            param_group = dict(names=[], params=[])
            if "lr" in param_dicts[i].keys():
                param_group["lr"] = param_dicts[i].lr
            if "momentum" in param_dicts[i].keys():
                param_group["momentum"] = param_dicts[i].momentum
            if "weight_decay" in param_dicts[i].keys():
                param_group["weight_decay"] = param_dicts[i].weight_decay
            cfg.params.append(param_group)

        for n, p in model.named_parameters():
            flag = False
            for i in range(len(param_dicts)):
                if param_dicts[i].keyword in n:
                    cfg.params[i + 1]["names"].append(n)
                    cfg.params[i + 1]["params"].append(p)
                    flag = True
                    break
            if not flag:
                cfg.params[0]["names"].append(n)
                cfg.params[0]["params"].append(p)

        log = RankedLogger(__name__, rank_zero_only=True)

        for i in range(len(cfg.params)):
            param_names = cfg.params[i].pop("names")
            message = ""
            for key in cfg.params[i].keys():
                if key != "params":
                    message += f" {key}: {cfg.params[i][key]};"
            log.info(f"Params Group {i+1} -{message} Params: {param_names}.")

    return OPTIMIZERS.build(cfg=OmegaConf.to_container(cfg, resolve=True))


def group_with_matcher(
    named_objects: Iterator[Tuple[str, Any]],
    group_matcher: Union[Dict, Callable],
    return_values: bool = False,
    reverse: bool = False,
):
    if isinstance(group_matcher, dict):
        # dictionary matcher contains a dict of raw-string regex expr that must be compiled
        compiled = []
        for group_ordinal, (group_name, mspec) in enumerate(group_matcher.items()):
            if mspec is None:
                continue
            # map all matching specifications into 3-tuple (compiled re, prefix, suffix)
            if isinstance(mspec, (tuple, list)):
                # multi-entry match specifications require each sub-spec to be a 2-tuple (re, suffix)
                for sspec in mspec:
                    compiled += [(re.compile(sspec[0]), (group_ordinal,), sspec[1])]
            else:
                compiled += [(re.compile(mspec), (group_ordinal,), None)]
        group_matcher = compiled

    def _get_grouping(name):
        if isinstance(group_matcher, (list, tuple)):
            for match_fn, prefix, suffix in group_matcher:
                r = match_fn.match(name)
                if r:
                    parts = (prefix, r.groups(), suffix)
                    # map all tuple elem to int for numeric sort, filter out None entries
                    return tuple(map(float, chain.from_iterable(filter(None, parts))))
            return (
                float("inf"),
            )  # un-matched layers (neck, head) mapped to largest ordinal
        else:
            ord = group_matcher(name)
            if not isinstance(ord, collections.abc.Iterable):
                return (ord,)
            return tuple(ord)

    # map layers into groups via ordinals (ints or tuples of ints) from matcher
    grouping = defaultdict(list)
    for k, v in named_objects:
        grouping[_get_grouping(k)].append(v if return_values else k)

    # remap to integers
    layer_id_to_param = defaultdict(list)
    lid = -1
    for k in sorted(filter(lambda x: x is not None, grouping.keys())):
        if lid < 0 or k[-1] != MATCH_PREV_GROUP[0]:
            lid += 1
        layer_id_to_param[lid].extend(grouping[k])

    if reverse:
        assert not return_values, "reverse mapping only sensible for name output"
        # output reverse mapping
        param_to_layer_id = {}
        for lid, lm in layer_id_to_param.items():
            for n in lm:
                param_to_layer_id[n] = lid
        return param_to_layer_id

    return layer_id_to_param


def group_parameters(
    module: nn.Module,
    group_matcher,
    return_values: bool = False,
    reverse: bool = False,
):
    return group_with_matcher(
        module.named_parameters(),
        group_matcher,
        return_values=return_values,
        reverse=reverse,
    )


def param_groups_weight_decay(
    model: nn.Module, weight_decay=1e-5, no_weight_decay_list=()
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def _group(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def _layer_map(model, layers_per_group=12, num_groups=None):
    def _in_head(n, hp):
        if not hp:
            return True
        elif isinstance(hp, (tuple, list)):
            return any([n.startswith(hpi) for hpi in hp])
        else:
            return n.startswith(hp)

    head_prefix = getattr(model, "pretrained_cfg", {}).get("classifier", None)
    names_trunk = []
    names_head = []
    for n, _ in model.named_parameters():
        names_head.append(n) if _in_head(n, head_prefix) else names_trunk.append(n)

    # group non-head layers
    num_trunk_layers = len(names_trunk)
    if num_groups is not None:
        layers_per_group = -(num_trunk_layers // -num_groups)
    names_trunk = list(_group(names_trunk, layers_per_group))

    num_trunk_groups = len(names_trunk)
    layer_map = {n: i for i, l in enumerate(names_trunk) for n in l}
    layer_map.update({n: num_trunk_groups for n in names_head})
    return layer_map


def param_groups_layer_decay(
    model: nn.Module,
    weight_decay: float = 0.05,
    no_weight_decay_list: Tuple[str] = (),
    layer_decay: float = 0.75,
    end_layer_decay: Optional[float] = None,
    verbose: bool = False,
):
    """
    Parameter groups for layer-wise lr decay & weight decay
    Based on BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    no_weight_decay_list = set(no_weight_decay_list)
    param_group_names = {}  # NOTE for debugging
    param_groups = {}

    if hasattr(model, "group_matcher"):
        # FIXME interface needs more work
        layer_map = group_parameters(
            model, model.group_matcher(coarse=False), reverse=True
        )
    else:
        # fallback
        layer_map = _layer_map(model)
    num_layers = max(layer_map.values()) + 1
    layer_max = num_layers - 1
    layer_scales = list(layer_decay ** (layer_max - i) for i in range(num_layers))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if param.ndim == 1 or name in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = layer_map.get(name, layer_max)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_groups:
            this_scale = layer_scales[layer_id]
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "param_names": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["param_names"].append(name)
        param_groups[group_name]["params"].append(param)

    if verbose:
        import json

        _logger.info("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def build_optimizer_v2(
    cfg,
    model_or_params,
    param_group_fn: Optional[Callable] = None,
    weight_decay: float = 0.0,
    **kwargs,
):
    if isinstance(model_or_params, nn.Module):
        # a model was passed in, extract parameters and add weight decays to appropriate layers
        no_weight_decay = {}
        if hasattr(model_or_params, "no_weight_decay"):
            no_weight_decay = model_or_params.no_weight_decay()

        if param_group_fn:
            parameters = param_group_fn(model_or_params)
        elif cfg.get("layer_decay") is not None:
            parameters = param_groups_layer_decay(
                model_or_params,
                weight_decay=cfg.get("weight_decay", 0.0),
                layer_decay=cfg.layer_decay,
                no_weight_decay_list=no_weight_decay,
                verbose=cfg.get("verbose", False),
            )
            weight_decay = 0.0
        elif cfg.get("weight_decay", 0.0) and cfg.get("filter_bias_and_bn", True):
            parameters = param_groups_weight_decay(
                model_or_params, cfg.get("weight_decay", 0.0), no_weight_decay
            )
            weight_decay = 0.0
        else:
            parameters = model_or_params.parameters()

    opt_args = dict(weight_decay=weight_decay, **kwargs)

    if cfg.get("lr") is not None:
        opt_args.setdefault("lr", cfg.lr)

    if cfg.get("foreach") is None:
        if cfg.type.lower() in _DEFAULT_FOREACH:
            opt_args.setdefault("foreach", True)
    else:
        opt_args["foreach"] = cfg.foreach

    opt_args["params"] = parameters
    opt_args["type"] = cfg.type

    return OPTIMIZERS.build(opt_args)


if __name__ == "__main__":
    from omegaconf import DictConfig
    from timm.models import resnet50

    optimizer = build_optimizer_v2(
        OmegaConf.to_container(
            DictConfig(
                dict(
                    type="SGD",
                    params=[],
                )
            ),
            resolve=True,
        ),
        resnet50(),
        layer_decay=0.1,
    )
    print(optimizer)
