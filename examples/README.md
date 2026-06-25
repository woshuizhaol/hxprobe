# Examples

```bash
python examples/quickstart.py     # score the bundled ubiquitin ensemble, compare to experiment
python examples/convergence.py    # show the ~50-conformer convergence plateau
```

Both use the bundled 50-conformer leakage-free ubiquitin ensemble
(`hxprobe.example_ensemble_path()`) and its experimental native-state opening
free energies (`hxprobe.load_experimental()`). No external data needed.
