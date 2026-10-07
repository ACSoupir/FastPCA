#!/usr/bin/env python
"""How accurate is the randomized FastPCA kernel at k=10 on this data?

    python TESTING/irlba_accuracy.py --x ... --m 978 --n 199112 \
        --ref-bin TESTING/_tmp/irlba_fast_F199112_na00 --qiters 2,4,6

Compares inst/python/torch_svd_func.py::randomized_svd_py against the near-exact
irlba reference written by TESTING/irlba_r.R --save-bin. Reports the relative
singular-value error and the principal angles between loading subspaces, once per
q_iter (each repeated --reps times, since the sketch is random).
"""
import argparse
import importlib.util
import json

import numpy as np


def principal_angles(A, B):
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def load_kernel(path):
    spec = importlib.util.spec_from_file_location("fastpca_torch_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--ref-bin", required=True)
    ap.add_argument("--kernel", default="inst/python/torch_svd_func.py")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--qiters", default="2,4,6")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    mod = load_kernel(args.kernel)

    raw = np.fromfile(args.x, dtype="<f8")
    A_np = raw.reshape((args.n, args.m), order="C").T.copy(order="F")
    A_np -= A_np.mean(axis=0)

    ref_d = np.sqrt(np.fromfile(args.ref_bin + ".eig.f64", dtype="<f8"))
    meta = json.load(open(args.ref_bin + ".json"))
    ref_v = np.fromfile(args.ref_bin + ".loadings.f64", dtype="<f8").reshape(
        (meta["nvar"], meta["ncomp"]), order="F")

    rows = []
    for q in [int(x) for x in args.qiters.split(",")]:
        errs, angs = [], []
        for _ in range(args.reps):
            _, S, Vh = mod.randomized_svd_py(A_np, k=args.k, p=args.p,
                                             q_iter=q, device="CPU",
                                             cores=args.threads)
            S = np.asarray(S)[:args.k]
            errs.append(float(np.max(np.abs(S - ref_d[:args.k]) / np.abs(ref_d[:args.k]))))
            angs.append(float(np.max(principal_angles(Vh.T[:, :args.k],
                                                       ref_v[:, :args.k]))))
        row = {"q_iter": q, "max_rel_sv_err_worst": max(errs),
               "max_rel_sv_err_best": min(errs),
               "max_angle_deg_worst": max(angs), "reps": args.reps}
        rows.append(row)
        print(f"q_iter={q}: max_rel_sv_err {min(errs):.4f}..{max(errs):.4f}  "
              f"max_angle<= {max(angs):.3f} deg", flush=True)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
    print("ACCURACY_JSON " + json.dumps(rows))


if __name__ == "__main__":
    main()
