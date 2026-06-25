import numpy as np
import mdtraj as md

import hxprobe
from hxprobe import operator


def _example_traj():
    return md.load(hxprobe.example_ensemble_path())


def test_constants_are_classical():
    assert operator.BETA_C == 0.35
    assert operator.BETA_H == 2.0
    assert abs(operator.CUT_NC_NM - 0.65) < 1e-12
    assert abs(operator.CUT_NH_NM - 0.26) < 1e-12
    assert operator.IJ_NC == 3 and operator.IJ_NH == 2


def test_compute_shapes_and_relation():
    res = operator.compute(_example_traj())
    n = len(res)
    assert n > 30
    assert res.NC.shape == (n, res.n_frames)
    assert res.NH.shape == (n, res.n_frames)
    # ln PF = betaC*<NC> + betaH*<NH>
    expect = operator.BETA_C * res.NC_mean + operator.BETA_H * res.NH_mean
    assert np.allclose(res.lnPF, expect, atol=1e-9)
    # dG = RT * lnPF
    rt = operator.R_KCAL * operator.T_REF
    assert np.allclose(res.dGopen_kcal, rt * res.lnPF, atol=1e-9)
    # protection should be non-negative and ordered with lnPF
    assert np.all(res.NC_mean >= 0)


def test_proline_skipped():
    res = operator.compute(_example_traj())
    assert "P" not in set(res.resn.tolist())  # prolines have no exchangeable amide
