"""Command-line interface for hxprobe."""
from __future__ import annotations

import argparse
import sys


def _add_common(p):
    p.add_argument("ensemble", help="multi-model PDB (.pdb/.pdb.gz) or trajectory file")
    p.add_argument("--top", default=None, help="topology file (for trajectory inputs)")
    p.add_argument("--protonate", default="auto",
                   choices=["auto", "none", "pdbfixer"],
                   help="how to obtain backbone amide hydrogens (default: auto)")
    p.add_argument("--betaC", type=float, default=None, help="contact coefficient")
    p.add_argument("--betaH", type=float, default=None, help="H-bond coefficient")
    p.add_argument("--temperature", type=float, default=None, help="temperature (K)")


def _operator_kw(args):
    from . import BETA_C, BETA_H, T_REF
    return dict(
        protonate=args.protonate,
        betaC=BETA_C if args.betaC is None else args.betaC,
        betaH=BETA_H if args.betaH is None else args.betaH,
        temperature=T_REF if args.temperature is None else args.temperature,
    )


def _cmd_score(args):
    from . import score_ensemble
    res = score_ensemble(args.ensemble, top=args.top, **_operator_kw(args))
    df = res.to_dataframe()
    if args.out:
        res.to_csv(args.out)
        print(f"wrote {len(res)} residues to {args.out}")
    else:
        print(df.to_string(index=False))
    return 0


def _cmd_converge(args):
    from . import convergence
    out = convergence(args.ensemble, top=args.top, **_operator_kw(args))
    try:
        print(out.to_string(index=False))
    except AttributeError:
        for row in out:
            print(row)
    return 0


def _cmd_example(args):
    from . import (convergence, example_ensemble_path, load_experimental,
                   score_ensemble, spearman)
    path = example_ensemble_path()
    exp = load_experimental()
    print(f"bundled example: 50-conformer leakage-free ubiquitin ensemble\n  {path}")
    res = score_ensemble(path, protonate="none")  # already protonated
    ref = [exp.get(int(rs)) for rs in res.resSeq]
    rho = spearman(res.lnPF, [r if r is not None else float("nan") for r in ref])
    n_overlap = sum(1 for r in ref if r is not None)
    print(f"\nper-residue ln PF vs experimental dG_open (native-state HX):")
    print(f"  Spearman rho = {rho:+.3f}  over {n_overlap} measured residues")
    print(f"\nconvergence with ensemble size:")
    conv = convergence(path, protonate="none", reference=exp)
    try:
        print(conv.to_string(index=False))
    except AttributeError:
        for row in conv:
            print("  ", row)
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="hxprobe",
        description="The 50-conformer probe: residue-resolved hydrogen-exchange "
                    "opening free energies from conformational ensembles.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("score", help="per-residue opening free energies for an ensemble")
    _add_common(s)
    s.add_argument("--out", default=None, help="write a CSV instead of printing")
    s.set_defaults(func=_cmd_score)

    c = sub.add_parser("converge", help="convergence of the readout with ensemble size")
    _add_common(c)
    c.set_defaults(func=_cmd_converge)

    e = sub.add_parser("example", help="run the bundled ubiquitin example")
    e.set_defaults(func=_cmd_example)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
