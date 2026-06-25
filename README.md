# hxprobe — the 50-conformer probe

**Read residue-resolved hydrogen-exchange opening free energies out of a conformational ensemble.**

`hxprobe` turns a conformational ensemble (from a generative model, molecular
dynamics, or any source) into per-residue protection factors and opening free
energies (ΔG_open, in kcal/mol) using a white-box, two-parameter physical
operator. It is an inexpensive, physically interpretable probe of how well an
ensemble reproduces the *near-equilibrium local opening* that hydrogen–deuterium
exchange measures — and it converges within roughly **50 conformers**, which is
where the name comes from.

```
ln PF_i = β_C · ⟨N_C,i⟩ + β_H · ⟨N_H,i⟩            (ensemble average)
ΔG_open,i = RT · ln PF_i                            (EX2 regime)
```

`N_C` counts heavy atoms near each backbone amide nitrogen and `N_H` counts the
amide's backbone hydrogen bonds, averaged over the ensemble. The two
coefficients are **fixed to their classical Best–Vendruscolo values** (0.35 and
2.0) and are *not* fitted to stability data, so any residue-level signal the
probe recovers comes from the ensemble, not from a tuned scoring function.

## Install

```bash
pip install hxprobe
```

This pulls in `numpy`, `pandas`, and `mdtraj`. Two optional extras:

```bash
pip install "hxprobe[fix]"   # PDBFixer/OpenMM: repair + protonate raw structures
pip install "hxprobe[diff]"  # PyTorch: differentiable operator for steering
```

## Quickstart

```python
import hxprobe

# Score the bundled 50-conformer ubiquitin ensemble (already protonated).
res = hxprobe.score_ensemble(hxprobe.example_ensemble_path(), protonate="none")
print(res.to_dataframe().head())          # resSeq, resn, NC_mean, NH_mean, lnPF, dGopen_kcal

# Compare to experimental native-state HX opening free energies.
exp = hxprobe.load_experimental()          # {resSeq: dG_open}
ref = [exp.get(int(r), float("nan")) for r in res.resSeq]
print("Spearman vs experiment:", round(hxprobe.spearman(res.lnPF, ref), 3))
```

Score *your own* ensemble — a multi-model PDB, or a trajectory plus topology:

```python
res = hxprobe.score_ensemble("my_ensemble.pdb")               # multi-model PDB
res = hxprobe.score_ensemble("traj.xtc", top="topology.pdb")  # trajectory + topology
res.to_csv("opening_free_energies.csv")
```

If your structures are raw heavy-atom coordinates without hydrogens, the H-bond
term is obtained either by a geometric amide-H placement (default, no extra
dependencies) or, more faithfully, with PDBFixer:

```python
res = hxprobe.score_ensemble("raw_heavy_atom.pdb", protonate="pdbfixer")  # needs hxprobe[fix]
```

## Command line

```bash
hxprobe example                       # run the bundled ubiquitin demo
hxprobe score my_ensemble.pdb         # print the per-residue table
hxprobe score traj.xtc --top top.pdb --out dG.csv
hxprobe converge my_ensemble.pdb      # show convergence with ensemble size
```

## What you get back

`score_ensemble` returns a `ProtectionResult` with NumPy arrays and a
`.to_dataframe()` / `.to_csv()` helper:

| field | meaning |
|---|---|
| `resSeq`, `resn` | residue number and one-letter code |
| `NC_mean`, `NH_mean` | ensemble-averaged contacts / hydrogen bonds |
| `lnPF`, `log10PF` | log protection factor |
| `dGopen_kcal` | opening free energy ΔG_open (kcal/mol) |

Two further entry points:

* **`convergence(ensemble)`** — Spearman correlation of the `n`-conformer
  readout against the full-ensemble readout (and, optionally, against an
  experimental reference), showing the plateau near ~50 conformers.
* **`global_unfolding(ensemble)`** — `RT · min_c ⟨ln PF⟩_residues`, the
  protection of the most-open conformer, a bounded proxy for global fold
  stability (the unfolded-state limit of the ensemble).

## How it works

For each backbone amide (prolines and non-standard residues are skipped):

* **`N_C`** — heavy atoms within **6.5 Å** of the amide nitrogen, sequence
  separation `|i − j| ≥ 3`, hydrogens excluded.
* **`N_H`** — backbone carbonyl oxygens within **2.6 Å** of the amide hydrogen,
  sequence separation `|i − j| ≥ 2`.

Counts are computed per conformer and **averaged over the ensemble before** the
linear combination is formed, so a residue that is buried in most conformers but
exposed in a rare open state receives the reduced mean contact count its
protection reflects. The contact term dominates, so the readout is robust even
when hydrogens are placed geometrically rather than with a full protonation step.

## Reproducing the bundled example

`hxprobe example` scores a 50-conformer leakage-free ubiquitin ensemble and
recovers the experimental native-state opening free energies at Spearman
ρ ≈ 0.58, with the correlation plateauing by ~25–50 conformers — the behaviour
that motivates the probe.

## Citing

If you use `hxprobe`, please cite the accompanying study on residue-resolved
hydrogen-exchange free energies as a benchmark for generative conformational
ensembles. (Reference to be added on publication.)

## License

MIT.
