"""Generate all evidence images for every scenario.

Primary: Gemini image API (verified working). Fallback per-image: Cosmos t2i via nano queue.
Parallel across images (Gemini side); Cosmos jobs serialize on the GPU queue anyway.

Usage:  python -m artifact.dataset.gen_evidence_images [--only S07] [--workers 6]
"""
import argparse, json, time
from concurrent.futures import ThreadPoolExecutor

from artifact.common import paths
from artifact.common.gemini_client import generate_image
from artifact.common.nano_queue import queue_generate
from artifact.dataset.safety_style import build_image_prompt


def cosmos_t2i(prompt, out_png):
    """Fallback: wrap prompt in a minimal Cosmos pos json and queue with num_frames=1."""
    pos = out_png.with_suffix(".cosmos_pos.json")
    json.dump({"scene_description": prompt}, open(pos, "w"))
    queue_generate(pos, out_png, num_frames=1, steps=35, timeout_s=900)
    pos.unlink(missing_ok=True)


def gen_one(sid, ev):
    d = paths.scenario_dir(sid)
    out = d / "evidence" / ev["file"]
    if out.exists() and out.stat().st_size > 10000:
        return ("skip", 0.0)
    prompt = build_image_prompt(ev["type"], ev["desc"])
    try:
        ok, cost = generate_image(prompt, out)
        if ok:
            return ("gemini", cost)
    except Exception as e:
        print(f"  {sid}/{ev['file']} gemini error: {str(e)[:80]}", flush=True)
        cost = 0.0
    try:
        cosmos_t2i(prompt, out)
        return ("cosmos", cost)
    except Exception as e:
        print(f"  {sid}/{ev['file']} FAILED both paths: {str(e)[:80]}", flush=True)
        return ("failed", cost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    jobs = []
    for sdir in sorted(paths.SCENARIOS.iterdir()):
        sj = sdir / "scenario.json"
        if not sj.exists() or (args.only and sdir.name != args.only):
            continue
        s = json.load(open(sj))
        for ev in s["evidence_images"]:
            jobs.append((s["id"], ev))
    print(f"{len(jobs)} images to generate", flush=True)

    t0, stats, total = time.time(), {}, 0.0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen_one, sid, ev): (sid, ev) for sid, ev in jobs}
        for i, f in enumerate(futs):
            src, cost = f.result()
            stats[src] = stats.get(src, 0) + 1
            total += cost
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(jobs)} done {stats} ${total:.2f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE {stats}, ${total:.2f}, {(time.time()-t0)/60:.1f} min", flush=True)
    # record provenance for the paper's implementation section
    json.dump(stats, open(paths.RESULTS / "evidence_image_stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
