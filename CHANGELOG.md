# Changelog

## 0.1.0

Initial release.

- White-box two-parameter Best–Vendruscolo forward operator
  (`ln PF = β_C·⟨N_C⟩ + β_H·⟨N_H⟩`, `ΔG_open = RT·ln PF`).
- `score_ensemble`, `convergence`, and `global_unfolding` high-level API.
- Geometric amide-H placement (NumPy only) with an optional PDBFixer/OpenMM
  backend (`hxprobe[fix]`) for raw heavy-atom structures.
- Optional differentiable operator (`hxprobe[diff]`).
- `hxprobe` command-line interface (`score`, `converge`, `example`).
- Bundled 50-conformer leakage-free ubiquitin ensemble with experimental
  native-state opening free energies.
