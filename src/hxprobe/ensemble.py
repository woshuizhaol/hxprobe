"""Loading conformational ensembles.

Thin wrappers over MDTraj that accept the common ways an ensemble is stored:
a multi-model PDB (optionally gzipped), or a trajectory file plus a topology.
"""
from __future__ import annotations

from typing import Optional


def load_ensemble(path: str, top: Optional[str] = None):
    """Load a conformational ensemble as an ``mdtraj.Trajectory``.

    Parameters
    ----------
    path : str
        A multi-model PDB (``.pdb`` / ``.pdb.gz``) or a trajectory file
        (``.xtc``, ``.dcd``, ``.h5`` ...).
    top : str, optional
        Topology file (e.g. a ``.pdb``); required for trajectory formats that
        do not embed topology.

    Returns
    -------
    mdtraj.Trajectory
    """
    import mdtraj as md

    if top is not None:
        return md.load(path, top=top)
    return md.load(path)


def optionally_protonate(traj, method: str = "auto"):
    """Return an ensemble guaranteed to be scorable for the H-bond term.

    ``method``:

    * ``"none"``   -- score as-is (geometric amide-H placement is used inside
      the operator when explicit hydrogens are absent).
    * ``"pdbfixer"`` -- repair missing heavy atoms and add real hydrogens with
      PDBFixer/OpenMM (the ``hxprobe[fix]`` extra). Most faithful for raw
      crystal or heavy-atom generated structures.
    * ``"auto"`` (default) -- use PDBFixer if it is installed and hydrogens are
      missing, otherwise fall back to ``"none"``.
    """
    from .protonate import has_explicit_hydrogens, pdbfixer_protonate

    if method == "none":
        return traj
    if has_explicit_hydrogens(traj):
        return traj
    if method == "pdbfixer":
        return pdbfixer_protonate(traj)
    if method == "auto":
        try:
            return pdbfixer_protonate(traj)
        except Exception:
            return traj
    raise ValueError(f"unknown protonation method: {method!r}")
