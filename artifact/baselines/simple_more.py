"""Two more zero-API simple baselines (all-local Cosmos rendering).

  caption-only:   photo captions only (no testimonies) -> Cosmos i2v (first scene photo)
  testimony-only: witness statements only (no photos)  -> Cosmos t2v

Usage: python -m artifact.baselines.simple_more --method caption-only [--ts night4]
Same run layout as cosmos_naive.py so judge/collect work unchanged.
"""
import argparse, json, subprocess, time, traceback

from artifact.common import paths
from artifact.common.nano_queue import queue_generate
from artifact.pipeline.m1_fuse import load_bundle


def build_text(method, captions, testimonies):
    if method == "caption-only":
        return "; ".join(c["caption"] for c in captions)[:1800]
    return " ".join(t["text"] for t in testimonies)[:1800]


def run(sid, method, ts):
    rd = paths.run_dir(sid, method, ts)
    man = {"sid": sid, "method": method, "brain": None, "ts": ts, "steps": {}, "cost_usd": 0.0}
    t0 = time.time()
    try:
        captions, testimonies, images = load_bundle(sid)
        text = build_text(method, captions, testimonies)
        fusion = {"narrative": text, "timeline": [], "gemini_rounds": 0}
        json.dump(fusion, open(rd / "fusion.json", "w"), indent=2, ensure_ascii=False)

        first_frame = None
        kb = "none"
        if method == "caption-only":
            scenes = [p for p in images if "scene" in p.name] or images
            if scenes:
                first_frame = scenes[0]
                kb = "evidence-photo"
                import shutil
                shutil.copy(first_frame, rd / "keyframe.png")
        man["steps"]["m1"] = {"cost": 0.0, "rounds": 0, "keyframe_backend": kb}
        man["steps"]["m2"] = {"cost": 0.0, "skipped": True}

        pos = rd / "render_pos.json"
        json.dump({"scene_description": text}, open(pos, "w"))
        out = rd / "video.mp4"
        queue_generate(pos, out, image=first_frame, num_frames=189, steps=35,
                       height=480, width=832, timeout_s=3600)
        man["steps"]["m3"] = {"backend": "cosmos", "cost_usd": 0.0}

        import imageio_ffmpeg
        frames = rd / "frames"
        frames.mkdir(exist_ok=True)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                        "-i", str(out), "-vf", "select='not(mod(n\\,24))',scale=640:-1",
                        "-vsync", "vfr", "-frames:v", "8", str(frames / "f%d.png")], check=True)
        man["status"] = "ok"
    except Exception as e:
        man["status"] = "error"
        man["error"] = str(e)[:400]
        traceback.print_exc()
    man["wall_seconds"] = round(time.time() - t0, 1)
    json.dump(man, open(rd / "manifest.json", "w"), indent=2)
    print(f"[{sid}/{method}] {man['status']} in {man['wall_seconds']:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["caption-only", "testimony-only"])
    ap.add_argument("--ts", default="night4")
    ap.add_argument("--scenes")
    a = ap.parse_args()
    sids = a.scenes.split(",") if a.scenes else [
        d.name for d in sorted(paths.SCENARIOS.iterdir())
        if (d / "scenario.json").exists()
        and json.load(open(d / "scenario.json"))["split"] == "main"]
    for sid in sids:
        run(sid, a.method, a.ts)


if __name__ == "__main__":
    main()
