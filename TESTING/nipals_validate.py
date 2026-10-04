#!/usr/bin/env python
"""Compare two NIPALS runs saved with --save-npz.

    python TESTING/nipals_validate.py --ref a.npz --test b.npz [--json out.json]

Reports relative eigenvalue error and principal angles between the loading
subspaces (sign- and scale-invariant, so no rotation alignment is needed).
"""
import argparse
import json

import numpy as np


def principal_angles(A, B):
    """Principal angles (degrees) between the column spaces of A and B."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None, help="reference .npz")
    ap.add_argument("--test", default=None, help="test .npz")
    ap.add_argument("--ref-bin", default=None,
                    help="reference prefix written by nipals_r.R --save-bin")
    ap.add_argument("--test-bin", default=None, help="test prefix as above")
    ap.add_argument("--m", type=int, default=0)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    def load(side, npz_path, bin_prefix):
        if npz_path:
            z = np.load(npz_path)
            return z["eig"], z["loadings"]
        if not bin_prefix:
            raise SystemExit(f"give --{side} or --{side}-bin")
        eig = np.fromfile(bin_prefix + ".eig.f64", dtype="<f8")
        with open(bin_prefix + ".json") as fh:
            meta = json.load(fh)
        nvar, nc = int(meta["nvar"]), int(meta["ncomp"])
        L = np.fromfile(bin_prefix + ".loadings.f64",
                        dtype="<f8").reshape((nvar, nc), order="F")
        return eig, L

    ea, La = load("ref", args.ref, args.ref_bin)
    eb, Lb = load("test", args.test, args.test_bin)
    n = min(len(ea), len(eb))
    rel = np.abs(ea[:n] - eb[:n]) / np.maximum(np.abs(ea[:n]), 1e-300)
    ang = principal_angles(La[:, :n], Lb[:, :n])
    out = {
        "n": int(n),
        "max_rel_eig_err": float(rel.max()),
        "mean_rel_eig_err": float(rel.mean()),
        "max_angle_deg": float(ang.max()),
        "mean_angle_deg": float(ang.mean()),
    }
    print(f"n={out['n']} max_rel_eig_err={out['max_rel_eig_err']:.3e} "
          f"mean={out['mean_rel_eig_err']:.3e} | max_angle={out['max_angle_deg']:.4f} deg "
          f"mean={out['mean_angle_deg']:.4f} deg")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh)
    print("VALIDATE_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
