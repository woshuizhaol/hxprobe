import numpy as np

import hxprobe


def test_reproduces_paper_correlation():
    """The bundled 50-conformer ubiquitin ensemble should recover the
    experimental opening free energies at Spearman rho ~ 0.58."""
    res = hxprobe.score_ensemble(hxprobe.example_ensemble_path(), protonate="none")
    exp = hxprobe.load_experimental()
    ref = [exp.get(int(r), np.nan) for r in res.resSeq]
    rho = hxprobe.spearman(res.lnPF, ref)
    assert 0.45 < rho < 0.70, f"unexpected rho={rho}"


def test_convergence_plateau():
    path = hxprobe.example_ensemble_path()
    exp = hxprobe.load_experimental()
    table = hxprobe.convergence(path, ns=[10, 25, 50], reference=exp, protonate="none")
    d = {int(r["n"]): r for _, r in table.iterrows()}
    # full ensemble correlates perfectly with itself
    assert d[50]["self_spearman"] > 0.999
    # 25 conformers already close to the 50-conformer reference readout
    assert d[25]["self_spearman"] > 0.9
    # experimental correlation is in the expected band at full size
    assert 0.45 < d[50]["ref_spearman"] < 0.70


def test_global_unfolding_runs():
    g = hxprobe.global_unfolding(hxprobe.example_ensemble_path(), protonate="none")
    assert np.isfinite(g)
    assert g > 0  # protection of the most-open conformer is positive on a folded domain


def test_spearman_matches_known_values():
    assert abs(hxprobe.spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9
    assert abs(hxprobe.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
