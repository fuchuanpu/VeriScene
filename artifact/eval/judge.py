"""LLM-as-judge scoring (Gemini, decoupled from the generation brain).

Five 1-5 dimensions against ground truth: ECR evidence coverage, FC factual consistency,
PP physical plausibility, TC temporal coherence, VF visual fidelity.
ECR additionally gets a programmatic component: fraction of evidence files cited in
fusion.json's evidence_refs.

  python -m artifact.eval.judge [--only S07] [--judge-model ...]
"""
import argparse, glob, json
from pathlib import Path

from artifact.common import paths
from artifact.common.gemini_client import upload, delete, generate_text, extract_json

RUBRIC = """You are scoring an AI system that reconstructs an incident from legal evidence.

GROUND TRUTH (what actually happened):
{gt}

SYSTEM RECONSTRUCTION (narrative + timeline):
{fusion}

The attached images are 8 chronological frames of the system's re-enactment video.

Evaluate as follows. Be strict.
- FC: go through EVERY ground-truth timeline event; classify each as reflected / contradicted / missing
  in the system narrative. Also judge whether the key_physical_fact is captured (true/false).
- TC: for each ADJACENT pair of ground-truth events, judge whether the frame sequence shows them in the
  correct order (count pairs shown in order vs total pairs judgeable; unjudgeable pairs count as wrong).
- PP: 1-5 (5 = no physically impossible motion in any frame).
- VF: 1-5 (5 = photorealistic coherent scene, no distortion).

Return ONLY JSON:
{{"FC_events": [{{"event":"<short>","status":"reflected|contradicted|missing"}}],
  "key_fact_captured": true/false,
  "TC_ordered": n, "TC_total": n,
  "PP": n, "VF": n,
  "rationale": {{"FC":"...","PP":"...","TC":"...","VF":"..."}}}}"""


def ecr_score(scenario, fusion):
    """Programmatic evidence coverage: cited files / all evidence files, 0-1."""
    all_files = {e["file"] for e in scenario["evidence_images"]} | \
                {t["file"] for t in scenario["testimonies"]}
    cited = set()
    for step in fusion.get("timeline", []):
        cited |= set(step.get("evidence_refs", []))
    return round(len(cited & all_files) / max(1, len(all_files)), 4)


def normalize(raw):
    """Judge v2 output -> five 0-1 decimal metrics (user requirement: decimal metrics)."""
    ev = raw.get("FC_events", [])
    reflected = sum(1 for e in ev if e.get("status") == "reflected")
    fc = reflected / max(1, len(ev))
    if raw.get("key_fact_captured") is False:      # missing the decisive fact caps FC
        fc = min(fc, 0.5)
    tc = raw.get("TC_ordered", 0) / max(1, raw.get("TC_total", 1))
    return {"FC": round(fc, 4), "TC": round(tc, 4),
            "PP": round(raw.get("PP", 0) / 5.0, 4), "VF": round(raw.get("VF", 0) / 5.0, 4)}


def judge_run(rd, judge_model=None, force=False):
    rd = Path(rd)
    if (rd / "scores.json").exists() and not force:
        return json.load(open(rd / "scores.json"))
    man = json.load(open(rd / "manifest.json"))
    if man.get("status") != "ok":
        return None
    sid = man["sid"]
    scenario = json.load(open(paths.scenario_dir(sid) / "scenario.json"))
    fusion = json.load(open(rd / "fusion.json"))

    from PIL import Image
    frames = sorted((rd / "frames").glob("f*.png"))
    # inline PIL images, not Files API: file-backed generate_content 403s under heavy use
    imgs = [Image.open(p) for p in frames]
    text, cost = generate_text(imgs + [RUBRIC.format(
        gt=json.dumps(scenario["ground_truth"], indent=1),
        fusion=json.dumps({k: fusion.get(k) for k in ("narrative", "timeline")}, indent=1)[:6000])],
        model=judge_model or paths.GEMINI_PRO)
    raw = extract_json(text)
    scores = normalize(raw)
    scores["ECR"] = ecr_score(scenario, fusion)
    scores["raw"] = raw
    scores["rationale"] = raw.get("rationale", {})
    scores["judge_cost"] = round(cost, 4)
    json.dump(scores, open(rd / "scores.json", "w"), indent=2, ensure_ascii=False)
    print(f"  scored {sid}/{man['method']}: ECR={scores.get('ECR')} FC={scores.get('FC')} PP={scores.get('PP')} "
          f"TC={scores.get('TC')} VF={scores.get('VF')}", flush=True)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--judge-model")
    ap.add_argument("--force", action="store_true", help="re-score runs that already have scores.json")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    from concurrent.futures import ThreadPoolExecutor
    pats = sorted(glob.glob(str(paths.SCENARIOS / (args.only or "S*") / "runs" / "*" / "*" / "manifest.json")))
    print(f"{len(pats)} runs to score", flush=True)

    def one(m):
        try:
            judge_run(Path(m).parent, judge_model=args.judge_model, force=args.force)
        except Exception as e:
            print(f"  !! {m}: {str(e)[:100]}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(one, pats))


if __name__ == "__main__":
    main()
