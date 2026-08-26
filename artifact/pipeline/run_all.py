"""Batch driver: run a method over scenarios, judging as runs complete.

  python -m artifact.pipeline.run_all --method ours [--brain opus] [--split main]
      [--scenes S01,S02] [--workers 3] [--ts batch1]

Gemini/Veo steps parallelize across scenes; Cosmos jobs serialize on the GPU queue.
"""
import argparse, json
from concurrent.futures import ThreadPoolExecutor

from artifact.common import paths
from artifact.eval.judge import judge_run
from artifact.pipeline.run_scene import run


def scene_ids(split, scenes):
    if scenes:
        return scenes.split(",")
    out = []
    for d in sorted(paths.SCENARIOS.iterdir()):
        sj = d / "scenario.json"
        if sj.exists() and json.load(open(sj))["split"] == split:
            out.append(d.name)
    return out


def one(sid, a):
    rd, man = run(sid, a.method, brain=a.brain, ts=a.ts, probe=not a.no_probe)
    if man.get("status") == "ok":
        try:
            judge_run(rd)
        except Exception as e:
            print(f"  judge failed {sid}: {str(e)[:80]}", flush=True)
    return man.get("status")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["ours", "e2e", "wo_fusion", "wo_reason"])
    ap.add_argument("--brain", default="opus", choices=list(paths.BRAINS))
    ap.add_argument("--split", default="main")
    ap.add_argument("--scenes")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--ts", default="batch")
    ap.add_argument("--no-probe", action="store_true")
    a = ap.parse_args()

    sids = scene_ids(a.split, a.scenes)
    print(f"method={a.method} brain={a.brain} over {len(sids)} scenes: {sids}", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        statuses = list(ex.map(lambda s: one(s, a), sids))
    print(f"BATCH DONE: {statuses.count('ok')}/{len(sids)} ok", flush=True)


if __name__ == "__main__":
    main()
