#!/usr/bin/env python
"""Merge RESULT_JSON lines + /usr/bin/time -l peak RSS into one results file.

    python TESTING/nipals_collect.py --jsonl a.jsonl --rss b.tsv --out results.json
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--rss", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rss = {}
    with open(args.rss) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[0]:
                rss[parts[0]] = int(parts[1] or 0)

    rows, seen_skip = [], []
    with open(args.jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if not line.startswith("{"):
                continue
            rec = json.loads(line)
            label_base = f"{rec['impl']}-{rec.get('tag','')}"
            rec["peak_rss_time_l_gb"] = round(rss.get(label_base, 0) / 1e9, 3)
            rec["label"] = label_base
            if rec.get("status") == "skipped":
                seen_skip.append(rec)
            else:
                rows.append(rec)

    hdr = f"{'label':<26} {'best_s':>8} {'peakRSS_GB':>11} {'iters_sum':>9}  eig1"
    print(hdr)
    for r in rows:
        it = sum(i for i in (r.get("iters") or []) if isinstance(i, int))
        e1 = r["eig"][0] if r.get("eig") else float("nan")
        print(f"{r['label']:<26} {r.get('best_s', float('nan')):>8.3f} "
              f"{r.get('peak_rss_time_l_gb', 0):>11.3f} {it:>9}  {e1:.6g}")
    for r in seen_skip:
        print(f"{r['label']:<26} SKIPPED: {r.get('reason','')}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rows": rows, "skipped": seen_skip}, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
