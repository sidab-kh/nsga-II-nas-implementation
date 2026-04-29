"""
core/fbnet.py
=============
Helpers for the FBNet search space used by HW-NAS-Bench.

HW-NAS-Bench stores FBNet costs as operator-level lookup tables. A full
architecture is defined by a fixed macro-architecture plus 22 searchable
positions, each choosing one of 9 candidate blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


FBNET_CHOICE_COUNT = 9
FBNET_SEARCH_DIM = 22
FBNET_SUPPORTED_DATASETS = ("cifar10", "cifar100")


@dataclass(frozen=True)
class FBNetStageSpec:
    h: int
    w: int
    cin: int
    cout: int
    stride: int


# Candidate order follows the commonly reported FBNet table:
# k3_e1, k3_e1_g2, k3_e3, k3_e6, k5_e1, k5_e1_g2, k5_e3, k5_e6, skip
_FBNET_CONV_CANDIDATES = (
    (1, 3, 1),
    (1, 3, 2),
    (3, 3, 1),
    (6, 3, 1),
    (1, 5, 1),
    (1, 5, 2),
    (3, 5, 1),
    (6, 5, 1),
)
FBNET_SKIP_CHOICE = FBNET_CHOICE_COUNT - 1


def _repeat_stage(
    start_h: int,
    start_cin: int,
    cout: int,
    blocks: int,
    first_stride: int,
) -> List[FBNetStageSpec]:
    specs = [
        FBNetStageSpec(
            h=start_h,
            w=start_h,
            cin=start_cin,
            cout=cout,
            stride=first_stride,
        )
    ]

    next_h = start_h if first_stride == 1 else start_h // 2
    for _ in range(blocks - 1):
        specs.append(
            FBNetStageSpec(
                h=next_h,
                w=next_h,
                cin=cout,
                cout=cout,
                stride=1,
            )
        )
    return specs


def get_fbnet_searchable_stages(dataset: str) -> List[FBNetStageSpec]:
    """
    Return the 22 searchable FBNet positions for the requested dataset.

    The CIFAR variants use the modified macro-architecture reported by
    HW-NAS-Bench. Accuracy labels are not provided for this search space.
    """
    if dataset not in FBNET_SUPPORTED_DATASETS:
        raise ValueError(
            f"FBNet is only supported for {FBNET_SUPPORTED_DATASETS}, "
            f"got {dataset!r}."
        )

    return (
        [FBNetStageSpec(h=32, w=32, cin=16, cout=16, stride=1)]
        + _repeat_stage(start_h=32, start_cin=16, cout=24, blocks=4, first_stride=1)
        + _repeat_stage(start_h=32, start_cin=24, cout=32, blocks=4, first_stride=2)
        + _repeat_stage(start_h=16, start_cin=32, cout=64, blocks=4, first_stride=2)
        + _repeat_stage(start_h=8, start_cin=64, cout=112, blocks=4, first_stride=1)
        + _repeat_stage(start_h=8, start_cin=112, cout=184, blocks=4, first_stride=2)
        + [FBNetStageSpec(h=4, w=4, cin=184, cout=352, stride=1)]
    )


def get_fbnet_fixed_block_keys(dataset: str) -> List[str]:
    """Return the fixed stem/head operator keys surrounding the 22 choices."""
    if dataset == "cifar10":
        fc_out = 10
    elif dataset == "cifar100":
        fc_out = 100
    else:
        raise ValueError(
            f"FBNet is only supported for {FBNET_SUPPORTED_DATASETS}, "
            f"got {dataset!r}."
        )

    return [
        "ConvNorm_H32_W32_Cin3_Cout16_kernel3_stride1_group1",
        "ConvNorm_H4_W4_Cin352_Cout1504_kernel1_stride1_group1",
        "AvgP_H4_W4_Cin1504_Cout1504_kernel4_stride1",
        f"FC_Cin1504_Cout{fc_out}",
    ]


def fbnet_choice_to_key(stage: FBNetStageSpec, choice_idx: int) -> str:
    if not 0 <= choice_idx < FBNET_CHOICE_COUNT:
        raise ValueError(
            f"FBNet choice must be in [0, {FBNET_CHOICE_COUNT - 1}], got {choice_idx}."
        )

    if choice_idx == FBNET_SKIP_CHOICE:
        return (
            f"Skip_H{stage.h}_W{stage.w}_Cin{stage.cin}_Cout{stage.cout}"
            f"_stride{stage.stride}"
        )

    exp, kernel, group = _FBNET_CONV_CANDIDATES[choice_idx]
    return (
        f"ConvBlock_H{stage.h}_W{stage.w}_Cin{stage.cin}_Cout{stage.cout}"
        f"_exp{exp}_kernel{kernel}_stride{stage.stride}_group{group}"
    )


def fbnet_architecture_to_keys(choices: Sequence[int], dataset: str) -> List[str]:
    stages = get_fbnet_searchable_stages(dataset)
    if len(choices) != len(stages):
        raise ValueError(
            f"FBNet architectures require {len(stages)} choices, got {len(choices)}."
        )

    keys = list(get_fbnet_fixed_block_keys(dataset))
    keys.extend(
        fbnet_choice_to_key(stage, int(choice))
        for stage, choice in zip(stages, choices)
    )
    return keys
