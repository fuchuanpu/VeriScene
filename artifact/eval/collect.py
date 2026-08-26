"""Aggregate every run's manifest.json + scores.json into data/results/all_scores.csv.

Columns: sid,split,case_type,method,brain,ts,ECR,FC,PP,TC,VF,cost_usd,wall_seconds,
         m1_cost,m2_cost,m3_backend,keyframe_backend,status
The export/ figure scripts read ONLY this file.
"""
import csv, glob, json
from pathlib import Path

from artifact.common import paths

COLS = ["sid", "split", "case_type", "method", "brain", "ts", "ECR", "FC", "PP", "TC", "VF",
        "cost_usd", "wall_seconds", "m1_cost", "m2_cost", "m3_backend", "keyframe_backend", "status"]


def main():
    rows = []
    for mp in sorted(glob.glob(str(paths.SCENARIOS / "S*" / "runs" / "*" / "*" / "manifest.json"))):
        rd = Path(mp).parent
        man = json.load(open(mp))
        sid = man["sid"]
        scen = json.load(open(paths.scenario_dir(sid) / "scenario.json"))
        row = {"sid": sid, "split": scen["split"], "case_type": scen["case_type"],
               "method": man["method"], "brain": man.get("brain"), "ts": man["ts"],
               "cost_usd": man.get("cost_usd"), "wall_seconds": man.get("wall_seconds"),
               "m1_cost": man.get("steps", {}).get("m1", {}).get("cost"),
               "m2_cost": man.get("steps", {}).get("m2", {}).get("cost"),
               "m3_backend": man.get("steps", {}).get("m3", {}).get("backend"),
               "keyframe_backend": man.get("steps", {}).get("m1", {}).get("keyframe_backend"),
               "status": man.get("status")}
        sp = rd / "scores.json"
        if sp.exists():
            sc = json.load(open(sp))
            row.update({k: sc.get(k) for k in ("ECR", "FC", "PP", "TC", "VF")})
        rows.append(row)

    out = paths.RESULTS / "all_scores.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    scored = sum(1 for r in rows if r.get("FC") is not None)
    print(f"{out}: {len(rows)} runs, {scored} scored", flush=True)


if __name__ == "__main__":
    main()
