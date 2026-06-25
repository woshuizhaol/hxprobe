"""hxprobe -- the 50-conformer probe.

Read residue-resolved hydrogen-exchange opening free energies out of a
conformational ensemble with a white-box, two-parameter physical operator.
"""
from .operator import (BETA_C, BETA_H, CUT_NC_NM, CUT_NH_NM, IJ_NC, IJ_NH,
                       R_KCAL, T_REF, ProtectionResult, compute, nc_nh_frame)
from .ensemble import load_ensemble, optionally_protonate
from .probe import (convergence, example_ensemble_path, global_unfolding,
                    load_experimental, score_ensemble, spearman)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "score_ensemble",
    "convergence",
    "global_unfolding",
    "compute",
    "nc_nh_frame",
    "ProtectionResult",
    "load_ensemble",
    "optionally_protonate",
    "example_ensemble_path",
    "load_experimental",
    "spearman",
    "BETA_C", "BETA_H", "CUT_NC_NM", "CUT_NH_NM", "IJ_NC", "IJ_NH",
    "R_KCAL", "T_REF",
]
