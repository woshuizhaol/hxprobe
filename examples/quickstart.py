"""Quickstart: score the bundled ubiquitin ensemble and compare to experiment.

Run with:  python examples/quickstart.py
"""
import hxprobe

# The bundled example is a 50-conformer leakage-free ubiquitin ensemble that has
# already been protonated, so we can score it directly (protonate="none").
path = hxprobe.example_ensemble_path()
print("scoring:", path)

res = hxprobe.score_ensemble(path, protonate="none")
df = res.to_dataframe()
print("\nfirst residues:")
print(df.head(8).to_string(index=False))

# Compare per-residue ln PF to experimental native-state opening free energies.
exp = hxprobe.load_experimental()
ref = [exp.get(int(r), float("nan")) for r in res.resSeq]
rho = hxprobe.spearman(res.lnPF, ref)
n = sum(1 for r in ref if r == r)  # count non-NaN
print(f"\nSpearman(ln PF, experimental dG_open) = {rho:+.3f}  over {n} residues")

# The global fold-stability proxy (most-open conformer).
print(f"global unfolding proxy (RT * min-conformer mean ln PF) = "
      f"{hxprobe.global_unfolding(path, protonate='none'):.2f} kcal/mol")
