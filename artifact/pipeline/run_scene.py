"""Single-scenario pipeline entry.

  python -m artifact.pipeline.run_scene --scene S12 --method ours [--brain opus] [--ts demo]

Methods: ours | e2e | wo_fusion | wo_reason
Writes data/scenarios/<sid>/runs/<method-brain>/<ts>/{fusion,reasoning,manifest}.json,
keyframe.png, video.mp4, frames/, and later scores.json (eval/judge.py).
"""
import argparse, json, time, traceback

from artifact.common import paths
from artifact.pipeline import m1_fuse, m2_reason, m3_render


def naive_spec(fusion):
    """Video spec without physical reasoning (w/o-reason ablation + e2e)."""
    return {"video_prompt": fusion["motion_hint"] + " " + fusion["narrative"][:400],
            "physics_constraints": [], "negative_terms": ["frozen scene", "static image"],
            "cost_usd": 0.0}


def run(sid, method, brain="opus", ts=None, probe=True):
    ts = ts or time.strftime("%m%d_%H%M")
    label = f"{method}-{brain}" if method == "ours" else method
    rd = paths.run_dir(sid, label, ts)
    man = {"sid": sid, "method": method, "brain": brain, "ts": ts, "steps": {}, "cost_usd": 0.0}
    t0 = time.time()
    try:
        if method == "e2e":
            from artifact.baselines.e2e_monolithic import e2e
            fusion, c1 = e2e(sid, rd)
        else:
            fusion, c1 = m1_fuse.fuse(sid, rd, brain=brain,
                                      single_shot=(method == "wo_fusion"))
        man["steps"]["m1"] = {"cost": round(c1, 4), "rounds": fusion.get("gemini_rounds", 1),
                              "keyframe_backend": fusion.get("keyframe_backend")}

        if method in ("ours", "wo_fusion"):
            spec, c2 = m2_reason.reason(sid, rd, fusion, brain=brain, probe=probe)
        else:
            spec, c2 = naive_spec(fusion), 0.0
            json.dump(spec, open(rd / "reasoning.json", "w"), indent=2)
        man["steps"]["m2"] = {"cost": round(c2, 4), "skipped": method in ("e2e", "wo_reason")}

        r3 = m3_render.render(sid, rd, spec)
        man["steps"]["m3"] = r3
        man["cost_usd"] = round(c1 + c2 + r3["cost_usd"], 4)
        man["status"] = "ok"
    except Exception as e:
        man["status"] = "error"
        man["error"] = str(e)[:500]
        traceback.print_exc()
    man["wall_seconds"] = round(time.time() - t0, 1)
    json.dump(man, open(rd / "manifest.json", "w"), indent=2)
    print(f"[{sid}/{label}] {man['status']} in {man['wall_seconds']:.0f}s, ${man['cost_usd']:.2f}", flush=True)
    return rd, man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--method", default="ours",
                    choices=["ours", "e2e", "wo_fusion", "wo_reason"])
    ap.add_argument("--brain", default="opus", choices=list(paths.BRAINS))
    ap.add_argument("--ts")
    ap.add_argument("--no-probe", action="store_true")
    a = ap.parse_args()
    run(a.scene, a.method, brain=a.brain, ts=a.ts, probe=not a.no_probe)
