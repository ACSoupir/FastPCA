#!/usr/bin/env python
"""Merge IRLBA benchmark outputs into one results file.

    python TESTING/irlba_collect.py --jsonl res.jsonl --rss rss.tsv \
        [--memguard mg.jsonl] --out results/irlba_bench_f.json

Inputs:
  jsonl     RESULT_JSON lines captured from each run (has impl + tag)
  rss       "label<TAB>bytes" rows scraped from /usr/bin/time -l
  memguard  MEMGUARD_JSON lines (the hard-cap supervisor's own peak)

The label is impl-tag, so a core sweep shows up as one row per setting.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--rss", default=None)
    ap.add_argument("--memguard", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rss, memg = {}, {}
    if args.rss:
        with open(args.rss) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2 and parts[0]:
                    rss[parts[0]] = int(parts[1] or 0)
    if args.memguard:
        with open(args.memguard) as fh:
            for line in fh:
                if "MEMGUARD_JSON" not in line:
                    continue
                try:
                    rec = json.loads(line.split("MEMGUARD_JSON", 1)[1].strip())
                except Exception:
                    continue
                if rec.get("memguard_label"):
                    memg[rec["memguard_label"]] = rec

    rows, skipped = [], []
    with open(args.jsonl) as fh:
        for line in fh:
            if "RESULT_JSON" not in line:
                continue
            try:
                rec = json.loads(line.split("RESULT_JSON", 1)[1].strip())
            except Exception:
                continue
            label = rec.get("label") or f"{rec.get('impl')}-{rec.get('tag','')}"
            rec["label"] = label
            mg = memg.get(label, {})
            rec["peak_rss_time_l_gb"] = round(rss.get(label, 0) / 1e9, 3)
            rec["memguard_peak_gb"] = mg.get("peak_tree_rss_gb")
            rec["killed_by_memguard"] = mg.get("killed", False)
            (skipped if rec.get("status") == "skipped" else rows).append(rec)

    hdr = (f"{'label':<30} {'best_s':>9} {'RSS_GB':>8} {'MG_GB':>7} "
           f"{'iter/mprod':>11}  sv1")
    print(hdr)
    for r in rows:
        it = r.get("iterations", "")
        mp = r.get("mprod", "")
        prog = f"{it}/{mp}" if it != "" else ""
        sv1 = r["sv"][0] if r.get("sv") else float("nan")
        print(f"{r['label']:<30} {r.get('best_s', float('nan')):>9.3f} "
              f"{r.get('peak_rss_time_l_gb', 0):>8.3f} "
              f"{(r.get('memguard_peak_gb') or 0):>7.3f} {prog:>11}  {sv1:.6g}")
    for r in skipped:
        print(f"{r['label']:<30} SKIPPED: {r.get('reason','')}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rows": rows, "skipped": skipped}, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
