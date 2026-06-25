"""High-level interface: the 50-conformer probe.

* :func:`score_ensemble` -- per-residue opening free energies for an ensemble.
* :func:`convergence`    -- how the readout converges with ensemble size.
* :func:`global_unfolding` -- the most-open-conformer global stability proxy.
* :func:`example_ensemble_path` / :func:`load_experimental` -- bundled data.
"""
from __future__ import annotations

from importlib import resources
from typing import Optional, Sequence, Union

import numpy as np

from .operator import (BETA_C, BETA_H, R_KCAL, T_REF, ProtectionResult,
                       _largest_protein_chain, compute)
from .ensemble import load_ensemble, optionally_protonate


# --------------------------------------------------------------------------- #
# numpy-only Spearman (avoids a SciPy dependency)
# --------------------------------------------------------------------------- #
def _rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    sx = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation (NumPy only)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra, rb = _rankdata(a[m]), _rankdata(b[m])
    return float(np.corrcoef(ra, rb)[0, 1])


def _as_traj(ensemble, top=None):
    if isinstance(ensemble, str):
        return load_ensemble(ensemble, top=top)
    return ensemble


def score_ensemble(ensemble, top: Optional[str] = None, protonate: str = "auto",
                   betaC: float = BETA_C, betaH: float = BETA_H,
                   temperature: float = T_REF, **kw) -> ProtectionResult:
    """Score a conformational ensemble into per-residue opening free energies.

    Parameters
    ----------
    ensemble : str or mdtraj.Trajectory
        Path to a multi-model PDB / trajectory, or a loaded trajectory.
    top : str, optional
        Topology file when ``ensemble`` is a trajectory path.
    protonate : {"auto", "none", "pdbfixer"}
        How to obtain backbone amide hydrogens for the H-bond term.  ``"auto"``
        uses explicit hydrogens if present, else PDBFixer if installed, else a
        geometric placement.  See :func:`hxprobe.ensemble.optionally_protonate`.
    """
    traj = _as_traj(ensemble, top)
    traj = _largest_protein_chain(traj)
    traj = optionally_protonate(traj, method=protonate)
    return compute(traj, betaC=betaC, betaH=betaH, temperature=temperature,
                   select_largest_chain=False, **kw)


def _lnpf_from_first_n(res: ProtectionResult, n: int, betaC: float, betaH: float):
    nc = np.nanmean(res.NC[:, :n], axis=1)
    nh = np.nanmean(res.NH[:, :n], axis=1)
    return betaC * nc + betaH * nh


def convergence(ensemble, ns: Optional[Sequence[int]] = None,
                reference: Optional[dict] = None, top: Optional[str] = None,
                protonate: str = "auto", betaC: float = BETA_C,
                betaH: float = BETA_H, **kw):
    """Convergence of the readout with ensemble size.

    Returns a list of dicts (and, if pandas is available, a DataFrame) with, for
    each ``n``: the Spearman correlation of the ``n``-conformer ln PF against
    the full-ensemble ln PF (self-convergence) and, if ``reference`` is given,
    against the experimental opening free energies.

    ``reference`` maps ``resSeq -> experimental dG_open`` (see
    :func:`load_experimental`).
    """
    res = score_ensemble(ensemble, top=top, protonate=protonate,
                         betaC=betaC, betaH=betaH, **kw)
    F = res.n_frames
    if ns is None:
        ns = [n for n in (5, 10, 25, 50, 100, 200) if n <= F] or [F]
        if F not in ns:
            ns = list(ns) + [F]
    full = res.lnPF
    ref_vec = None
    if reference is not None:
        ref_vec = np.array([reference.get(int(rs), np.nan) for rs in res.resSeq], float)

    rows = []
    for n in ns:
        n = int(min(n, F))
        lnpf_n = _lnpf_from_first_n(res, n, betaC, betaH)
        row = {"n": n, "self_spearman": spearman(lnpf_n, full)}
        if ref_vec is not None:
            row["ref_spearman"] = spearman(lnpf_n, ref_vec)
        rows.append(row)
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except Exception:
        return rows


def global_unfolding(ensemble, top: Optional[str] = None, protonate: str = "auto",
                     betaC: float = BETA_C, betaH: float = BETA_H,
                     temperature: float = T_REF, **kw) -> float:
    """Most-open-conformer protection: a global fold-stability proxy.

    For each conformer the mean log-protection over residues is computed; the
    observable is ``RT * min_c <ln PF>_residues``, i.e. the protection of the
    single most-open conformer, taken as a proxy for the unfolded state.
    Larger values track higher global stability (dG_fold).
    """
    res = score_ensemble(ensemble, top=top, protonate=protonate,
                         betaC=betaC, betaH=betaH, temperature=temperature, **kw)
    per_conf_lnpf = betaC * res.NC + betaH * res.NH        # [n_res, n_frames]
    g = np.nanmean(per_conf_lnpf, axis=0)                  # per-conformer mean
    return float(R_KCAL * temperature * np.nanmin(g))


# --------------------------------------------------------------------------- #
# bundled example data (a 50-conformer leakage-free ubiquitin ensemble)
# --------------------------------------------------------------------------- #
def example_ensemble_path() -> str:
    """Path to the bundled 50-conformer ubiquitin ensemble (multi-model PDB)."""
    return str(resources.files("hxprobe").joinpath("data/ubiquitin_ensemble.pdb.gz"))


def load_experimental(path: Optional[str] = None) -> dict:
    """Load experimental per-residue opening free energies as ``{resSeq: dG}``.

    With no argument, returns the bundled ubiquitin native-state HX dataset.
    """
    if path is None:
        path = str(resources.files("hxprobe").joinpath("data/ubiquitin_dgopen.csv"))
    import csv
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out[int(row["resi"])] = float(row["dGopen_kcal"])
    return out
