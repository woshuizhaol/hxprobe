"""The white-box Best--Vendruscolo forward operator.

For each backbone amide *i* the log protection factor is the ensemble average

    ln PF_i = beta_C * <N_C,i> + beta_H * <N_H,i>

where ``N_C`` is the number of heavy atoms within ``cut_Nc`` of the amide
nitrogen (sequence separation >= ``ij_Nc``) and ``N_H`` is the number of
backbone carbonyl oxygens within ``cut_Nh`` of the amide hydrogen (sequence
separation >= ``ij_Nh``).  Under the EX2 regime the protection factor converts
to a per-residue opening free energy

    dG_open,i = RT * ln PF_i.

The two coefficients are fixed to their classical Best--Vendruscolo values
(0.35 and 2.0) and are *not* fitted to stability data, so any residue-level
signal the operator recovers originates in the ensemble itself.

This is a faithful, dependency-light re-implementation of the operator used in
the accompanying study; geometry is computed with MDTraj and NumPy only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Best--Vendruscolo classical coefficients (never fitted to stability labels).
BETA_C = 0.35
BETA_H = 2.0
# Geometric cut-offs in nanometres (6.5 Angstrom contacts, 2.6 Angstrom H-bond).
CUT_NC_NM = 0.65
CUT_NH_NM = 0.26
IJ_NC = 3
IJ_NH = 2
# Gas constant (kcal / mol / K) and reference temperature.
R_KCAL = 0.0019872041
T_REF = 298.15

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
_AMIDE_H_NAMES = ("H", "HN", "H1", "HT1")
_N_H_BOND_NM = 0.101  # ~1.01 Angstrom N-H bond length, for geometric placement.


@dataclass
class ProtectionResult:
    """Per-residue protection factors and opening free energies for an ensemble."""

    resSeq: np.ndarray           # residue numbers (author/topology numbering)
    resn: np.ndarray             # one-letter residue names
    NC_mean: np.ndarray          # ensemble-averaged heavy-atom contacts
    NH_mean: np.ndarray          # ensemble-averaged amide hydrogen bonds
    lnPF: np.ndarray             # ln protection factor
    dGopen_kcal: np.ndarray      # opening free energy (kcal/mol)
    NC: np.ndarray = field(repr=False)   # per-conformer contacts  [n_res, n_frames]
    NH: np.ndarray = field(repr=False)   # per-conformer H-bonds   [n_res, n_frames]
    weights: np.ndarray = field(repr=False)
    temperature: float = T_REF
    n_frames: int = 0

    @property
    def log10PF(self) -> np.ndarray:
        return self.lnPF / np.log(10.0)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame({
            "resSeq": self.resSeq,
            "resn": self.resn,
            "NC_mean": self.NC_mean,
            "NH_mean": self.NH_mean,
            "lnPF": self.lnPF,
            "log10PF": self.log10PF,
            "dGopen_kcal": self.dGopen_kcal,
        })

    def to_csv(self, path: str) -> None:
        self.to_dataframe().to_csv(path, index=False)

    def __len__(self) -> int:
        return len(self.resSeq)


def _largest_protein_chain(traj):
    """Atom-slice ``traj`` down to the chain with the most standard residues."""
    top = traj.topology
    best, best_n = None, -1
    for ch in top.chains:
        n = sum(1 for r in ch.residues if r.name in _THREE_TO_ONE)
        if n > best_n:
            best, best_n = ch, n
    if best is None:
        return traj
    return traj.atom_slice([a.index for a in best.atoms])


def _geometric_amide_h(xyz, top):
    """Geometric backbone amide-H position per residue, in the residue plane.

    Used only when explicit hydrogens are absent.  H is placed 1.01 A from N
    along the external bisector of the C(prev)-N-CA angle, the standard sp2
    amide placement.  Returns ``{residue.index: h_xyz_nm}``.
    """
    out = {}
    residues = list(top.residues)
    for r in residues:
        if r.name == "PRO" or r.name not in _THREE_TO_ONE:
            continue
        atoms = {a.name: a.index for a in r.atoms}
        if "N" not in atoms or "CA" not in atoms:
            continue
        prev_c = None
        if r.index > 0:
            pr = residues[r.index - 1]
            if pr.chain.index == r.chain.index:
                prev_c = next((a.index for a in pr.atoms if a.name == "C"), None)
        if prev_c is None:
            continue
        n = xyz[atoms["N"]]
        u_ca = xyz[atoms["CA"]] - n
        u_c = xyz[prev_c] - n
        nu_ca = np.linalg.norm(u_ca)
        nu_c = np.linalg.norm(u_c)
        if nu_ca < 1e-6 or nu_c < 1e-6:
            continue
        bis = u_ca / nu_ca + u_c / nu_c
        nb = np.linalg.norm(bis)
        if nb < 1e-6:
            continue
        out[r.index] = n - _N_H_BOND_NM * (bis / nb)
    return out


def nc_nh_frame(traj_single, cut_Nc=CUT_NC_NM, cut_Nh=CUT_NH_NM,
                ij_Nc=IJ_NC, ij_Nh=IJ_NH, place_h_if_missing=True):
    """Per-residue (N_C, N_H) for a single-frame MDTraj trajectory.

    Returns ``{resSeq: (one_letter, N_C, N_H)}``.  Prolines and non-standard
    residues are skipped (no exchangeable backbone amide).
    """
    top = traj_single.topology
    xyz = traj_single.xyz[0]
    heavy = np.array([a.index for a in top.atoms
                      if a.element is not None and a.element.symbol != "H"])
    hv_res = np.array([top.atom(i).residue.index for i in heavy])
    o_idx = np.array([a.index for a in top.atoms if a.name == "O"])
    o_res = np.array([top.atom(i).residue.index for i in o_idx])

    has_explicit_h = any(a.name in _AMIDE_H_NAMES and a.element is not None
                         and a.element.symbol == "H" for a in top.atoms)
    geo_h = {} if has_explicit_h else (
        _geometric_amide_h(xyz, top) if place_h_if_missing else {})

    out = {}
    for res in top.residues:
        if res.name == "PRO" or res.name not in _THREE_TO_ONE:
            continue
        ns = [a for a in res.atoms if a.name == "N"]
        if not ns:
            continue
        ri = res.index
        npos = xyz[ns[0].index]
        if heavy.size:
            d = np.linalg.norm(xyz[heavy] - npos, axis=1)
            nc = int(((d < cut_Nc) & (np.abs(hv_res - ri) >= ij_Nc)).sum())
        else:
            nc = 0

        h_pos = None
        hs = [a for a in res.atoms if a.name in _AMIDE_H_NAMES]
        if hs:
            h_pos = xyz[hs[0].index]
        elif ri in geo_h:
            h_pos = geo_h[ri]

        nh = 0
        if h_pos is not None and o_idx.size:
            dh = np.linalg.norm(xyz[o_idx] - h_pos, axis=1)
            nh = int(((dh < cut_Nh) & (np.abs(o_res - ri) >= ij_Nh)).sum())

        out[res.resSeq] = (_THREE_TO_ONE[res.name], nc, nh)
    return out


def compute(traj, weights=None, betaC=BETA_C, betaH=BETA_H,
            cut_Nc=CUT_NC_NM, cut_Nh=CUT_NH_NM, ij_Nc=IJ_NC, ij_Nh=IJ_NH,
            temperature=T_REF, select_largest_chain=True,
            place_h_if_missing=True) -> ProtectionResult:
    """Score a conformational ensemble into per-residue opening free energies.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The conformational ensemble (one or more frames).
    weights : array-like, optional
        Per-conformer Boltzmann weights (defaults to uniform).
    betaC, betaH : float
        Operator coefficients (default to the classical 0.35 / 2.0).
    temperature : float
        Temperature (K) for the RT * ln PF conversion.
    """
    if select_largest_chain:
        traj = _largest_protein_chain(traj)
    F = traj.n_frames
    if weights is None:
        w = np.ones(F) / F
    else:
        w = np.asarray(weights, float)
        w = w / w.sum()

    perframe = [nc_nh_frame(traj[k], cut_Nc, cut_Nh, ij_Nc, ij_Nh,
                            place_h_if_missing) for k in range(F)]
    all_res = sorted(set().union(*[set(d) for d in perframe])) if perframe else []

    resSeq, resn = [], []
    NCm, NHm = [], []
    for rs in all_res:
        ncv = np.array([perframe[k][rs][1] if rs in perframe[k] else np.nan
                        for k in range(F)], float)
        nhv = np.array([perframe[k][rs][2] if rs in perframe[k] else np.nan
                        for k in range(F)], float)
        name = next(perframe[k][rs][0] for k in range(F) if rs in perframe[k])
        resSeq.append(rs)
        resn.append(name)
        NCm.append(ncv)
        NHm.append(nhv)

    NC = np.array(NCm) if NCm else np.zeros((0, F))
    NH = np.array(NHm) if NHm else np.zeros((0, F))

    # weighted ensemble average over the frames in which each residue is present
    nc_mean = np.zeros(len(all_res))
    nh_mean = np.zeros(len(all_res))
    for i in range(len(all_res)):
        m = ~np.isnan(NC[i])
        ww = w[m] / w[m].sum() if m.any() else w
        nc_mean[i] = np.sum(ww * NC[i][m]) if m.any() else np.nan
        nh_mean[i] = np.sum(ww * NH[i][m]) if m.any() else np.nan

    lnPF = betaC * nc_mean + betaH * nh_mean
    dG = R_KCAL * temperature * lnPF
    return ProtectionResult(
        resSeq=np.array(resSeq), resn=np.array(resn),
        NC_mean=nc_mean, NH_mean=nh_mean, lnPF=lnPF, dGopen_kcal=dG,
        NC=NC, NH=NH, weights=w, temperature=temperature, n_frames=F,
    )
