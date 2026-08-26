"""Client for the resident Cosmos3-Nano queue server (agent_refine/servers/resident_nano_server.py).

t2i: num_frames=1 -> PNG. i2v/t2v: num_frames=189 -> 7s mp4 (~5 min @35 steps).
"""
import json, time
from pathlib import Path

from . import paths

PEND = paths.NANO_QUEUE / "pending"


def queue_generate(pos_json_path, out_path, neg_json_path=None, image=None,
                   num_frames=189, steps=35, guidance=6.0, shift=10.0, seed=1234,
                   timeout_s=3600, height=720, width=1280):
    """Submit one job and block until the output exists. Adapted from agent_refine_queue."""
    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size > 10000:
        print(f"    reusing existing {out_path.name}", flush=True)
        return
    job = {"prompt": str(pos_json_path), "negative": str(neg_json_path) if neg_json_path else None,
           "out": str(out_path), "image": str(image) if image else None,
           "num_frames": num_frames, "steps": steps, "guidance": guidance,
           "shift": shift, "seed": seed, "height": height, "width": width}
    jf = PEND / f"law_{out_path.stem}_{int(time.time())}.json"
    json.dump(job, open(jf, "w"), indent=1)
    t0 = time.time()
    while True:
        if out_path.exists() and out_path.stat().st_size > 0:
            time.sleep(2)  # let the writer flush
            return
        if not jf.exists() and not out_path.exists():
            raise RuntimeError(f"nano queue job failed (see server.log) for {out_path.name}")
        if time.time() - t0 > timeout_s:
            jf.unlink(missing_ok=True)
            raise RuntimeError(f"nano queue timeout for {out_path.name}")
        time.sleep(5)
