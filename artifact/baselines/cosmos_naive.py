"""Naive prompt-Cosmos baselines — the E2E family's dumb end. All local rendering.

  raw-cosmos:   concat captions+testimonies -> Cosmos i2v (first scene photo as frame 0)
  t2v-cosmos:   same text -> Cosmos t2v (no starting frame at all)
  flash-cosmos: one cheap text-only LLM call writes the video prompt -> Cosmos i2v

Usage: python -m artifact.baselines.cosmos_naive --method raw-cosmos [--scenes S01,..] [--ts night3]
Writes standard run layout (fusion.json/keyframe.png/video.mp4/frames/manifest.json)
so eval/judge.py and eval/collect.py work unchanged.
"""
import argparse, json, shutil, subprocess, time, traceback

from artifact.common import paths
from artifact.common.gemini_client import generate_text, extract_json
from artifact.common.nano_queue import queue_generate
from artifact.pipeline.m1_fuse import load_bundle

FLASH_PROMPT = """Here are captions of evidence photos and witness statements of one incident.
CAPTIONS: {captions}
STATEMENTS: {testimonies}
Return ONLY JSON: {{"narrative":"<what happened, 5 sentences>",
"video_prompt":"<one paragraph describing an 8-second re-enactment video of the incident>"}}"""


def build_text(captions, testimonies):
    cap = "; ".join(c["caption"] for c in captions)
    tst = " ".join(t["text"] for t in testimonies)
    return (cap + ". " + tst)[:1800]


def run(sid, method, ts):
    rd = paths.run_dir(sid, method, ts)
    man = {"sid": sid, "method": method, "brain": None, "ts": ts, "steps": {}, "cost_usd": 0.0}
    t0 = time.time()
    try:
        captions, testimonies, images = load_bundle(sid)
        cost = 0.0
        if method == "flash-cosmos":
            text, cost = generate_text([FLASH_PROMPT.format(
                captions=json.dumps(captions)[:2500],
                testimonies=" | ".join(t["text"] for t in testimonies)[:4000])],
                model=paths.GEMINI_FLASH)
            obj = extract_json(text)
            narrative, vprompt = obj.get("narrative", ""), obj.get("video_prompt", "")
        else:
            narrative = " ".join(t["text"] for t in testimonies)[:1200]
            vprompt = build_text(captions, testimonies)

        fusion = {"narrative": narrative, "timeline": [], "contradictions": [],
                  "keyframe_prompt": None, "motion_hint": vprompt[:200],
                  "gemini_rounds": 0, "cost_usd": round(cost, 4)}
        json.dump(fusion, open(rd / "fusion.json", "w"), indent=2, ensure_ascii=False)

        key = None
        if method != "t2v-cosmos":
            scene_imgs = [p for c, p in zip(captions, images) if c["type"] == "scene"] or images
            key = rd / "keyframe.png"
            shutil.copy(scene_imgs[0], key)
        man["steps"]["m1"] = {"cost": round(cost, 4), "rounds": 0,
                              "keyframe_backend": "evidence-photo" if key else "none"}

        pos = rd / "render_pos.json"
        json.dump({"scene_description": vprompt}, open(pos, "w"))
        out = rd / "video.mp4"
        queue_generate(pos, out, image=key, num_frames=189, steps=35,
                       height=480, width=832, timeout_s=3600)
        man["steps"]["m2"] = {"cost": 0.0, "skipped": True}
        man["steps"]["m3"] = {"backend": "cosmos", "cost_usd": 0.0}

        import imageio_ffmpeg
        frames = rd / "frames"
        frames.mkdir(exist_ok=True)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                        "-i", str(out), "-vf", "select='not(mod(n\\,24))',scale=640:-1",
                        "-vsync", "vfr", "-frames:v", "8", str(frames / "f%d.png")], check=True)
        man["cost_usd"] = round(cost, 4)
        man["status"] = "ok"
    except Exception as e:
        man["status"] = "error"
        man["error"] = str(e)[:400]
        traceback.print_exc()
    man["wall_seconds"] = round(time.time() - t0, 1)
    json.dump(man, open(rd / "manifest.json", "w"), indent=2)
    print(f"[{sid}/{method}] {man['status']} in {man['wall_seconds']:.0f}s", flush=True)
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["raw-cosmos", "t2v-cosmos", "flash-cosmos"])
    ap.add_argument("--scenes")
    ap.add_argument("--ts", default="night3")
    a = ap.parse_args()
    if a.scenes:
        sids = a.scenes.split(",")
    else:
        sids = [d.name for d in sorted(paths.SCENARIOS.iterdir())
                if (d / "scenario.json").exists()
                and json.load(open(d / "scenario.json"))["split"] == "main"]
    ok = sum(run(s, a.method, a.ts)["status"] == "ok" for s in sids)
    print(f"BATCH DONE {a.method}: {ok}/{len(sids)} ok", flush=True)


if __name__ == "__main__":
    main()
