"""Protonation / repair backends.

The contact term ``N_C`` needs no hydrogens, but the hydrogen-bond term
``N_H`` needs a backbone amide hydrogen.  Three options are available:

* explicit hydrogens already in the ensemble -- used directly;
* geometric placement inside the operator (NumPy only, no extra deps);
* PDBFixer/OpenMM repair + real hydrogen addition (``hxprobe[fix]``), which
  also rebuilds missing heavy atoms and is the most faithful for raw crystal
  or heavy-atom generated structures.
"""
from __future__ import annotations

_AMIDE_H_NAMES = ("H", "HN", "H1", "HT1")


def has_explicit_hydrogens(traj) -> bool:
    """True if the topology contains backbone amide hydrogens."""
    for a in traj.topology.atoms:
        if a.name in _AMIDE_H_NAMES and a.element is not None and a.element.symbol == "H":
            return True
    return False


def _fix_one_frame(traj1, add_missing_atoms=True, ph=7.0):
    import os
    import tempfile
    import mdtraj as md
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmp = tempfile.mktemp(suffix=".pdb", dir=shm)
    traj1.save_pdb(tmp)
    fixer = PDBFixer(filename=tmp)
    fixer.findMissingResidues()
    fixer.missingResidues = {}              # do not model whole missing residues
    fixer.findNonstandardResidues()
    fixer.findMissingAtoms()
    if not add_missing_atoms:
        fixer.missingAtoms = {}
        fixer.missingTerminals = {}
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    out = tempfile.mktemp(suffix=".pdb", dir=shm)
    with open(out, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh)
    fixed = md.load(out)
    os.remove(tmp)
    os.remove(out)
    return fixed


def pdbfixer_protonate(traj, add_missing_atoms=True, ph=7.0):
    """Repair + protonate every frame with PDBFixer; return a single trajectory.

    Requires the ``hxprobe[fix]`` extra (``pdbfixer``, ``openmm``).  Frames whose
    repaired atom count differs from the modal count are dropped so the result
    can be stacked into one trajectory.
    """
    import numpy as np
    import mdtraj as md

    fixed = [_fix_one_frame(traj[k], add_missing_atoms, ph)
             for k in range(traj.n_frames)]
    counts = [f.n_atoms for f in fixed]
    modal = max(set(counts), key=counts.count)
    keep = [f for f in fixed if f.n_atoms == modal]
    xyz = np.concatenate([f.xyz for f in keep], axis=0)
    return md.Trajectory(xyz, keep[0].topology)
