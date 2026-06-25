"""Show that the readout converges within ~50 conformers.

Run with:  python examples/convergence.py
"""
import hxprobe

path = hxprobe.example_ensemble_path()
exp = hxprobe.load_experimental()

# self_spearman: n-conformer ln PF vs full-ensemble ln PF (no experiment needed).
# ref_spearman:  n-conformer ln PF vs experimental dG_open.
table = hxprobe.convergence(path, ns=[5, 10, 25, 50], reference=exp,
                            protonate="none")
print(table.to_string(index=False))
print("\nThe correlation plateaus by ~25-50 conformers: ~50 conformers suffice.")
