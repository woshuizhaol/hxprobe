"""Differentiable Best--Vendruscolo operator (optional, requires PyTorch).

The hard contact and hydrogen-bond counts are replaced by smooth sigmoidal
switching functions controlled by a temperature ``tau``; the discrete operator
is recovered as ``tau -> 0``.  Because protection then becomes a differentiable
function of atomic coordinates, the residue-level readout can in principle
provide gradients to steer a generator toward the rare openings it
under-populates.

Install with ``pip install hxprobe[diff]``.
"""
from __future__ import annotations

from .operator import BETA_C, BETA_H, CUT_NC_NM, CUT_NH_NM, IJ_NC, IJ_NH, R_KCAL, T_REF


def soft_nc_nh(xyz, amideN_idx, amideH_idx, heavy_idx, O_idx, resid,
               cut_Nc=CUT_NC_NM, cut_Nh=CUT_NH_NM, ij_Nc=IJ_NC, ij_Nh=IJ_NH,
               tau=0.02):
    """Differentiable per-residue (N_C, N_H) for one conformer.

    Parameters
    ----------
    xyz : torch.Tensor ``[n_atoms, 3]`` (nanometres, requires_grad as needed)
    amideN_idx, amideH_idx : list[int]
        Per-residue amide N / amide H atom indices (``-1`` if absent).
    heavy_idx, O_idx : list[int]
        Heavy-atom and backbone-carbonyl-oxygen atom indices.
    resid : sequence[int]
        Residue index of every atom (used for the sequence-separation mask).
    tau : float
        Switching temperature; smaller is sharper (recovers the hard operator).

    Returns
    -------
    (NC, NH) : torch.Tensor, torch.Tensor    each ``[n_res]``
    """
    import torch

    dev = xyz.device
    R = len(amideN_idx)
    NC = torch.zeros(R, device=dev)
    NH = torch.zeros(R, device=dev)
    heavy = torch.as_tensor(heavy_idx, device=dev, dtype=torch.long)
    hres = torch.as_tensor([resid[i] for i in heavy_idx], device=dev)
    Os = torch.as_tensor(O_idx, device=dev, dtype=torch.long)
    Ores = torch.as_tensor([resid[i] for i in O_idx], device=dev)
    for r in range(R):
        ni = amideN_idx[r]
        if ni < 0:
            continue
        ri = resid[ni]
        d = torch.norm(xyz[heavy] - xyz[ni], dim=1)
        mask_c = (torch.abs(hres - ri) >= ij_Nc).float()
        NC[r] = torch.sum(torch.sigmoid((cut_Nc - d) / tau) * mask_c)
        hi = amideH_idx[r]
        if hi >= 0:
            dh = torch.norm(xyz[Os] - xyz[hi], dim=1)
            mask_h = (torch.abs(Ores - ri) >= ij_Nh).float()
            NH[r] = torch.sum(torch.sigmoid((cut_Nh - dh) / tau) * mask_h)
    return NC, NH


def soft_lnpf(xyz, amideN_idx, amideH_idx, heavy_idx, O_idx, resid,
              betaC=BETA_C, betaH=BETA_H, **kw):
    """Differentiable per-residue ln PF for one conformer."""
    NC, NH = soft_nc_nh(xyz, amideN_idx, amideH_idx, heavy_idx, O_idx, resid, **kw)
    return betaC * NC + betaH * NH
