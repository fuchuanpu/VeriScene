"""E2E baseline — everything dumped into Gemini in one shot, no agent orchestration.

Single call: all evidence images + testimonies -> narrative + keyframe prompt; then one
keyframe image and the same rendering backend as ours (isolating the agent's contribution).
"""
import json
from pathlib import Path

from artifact.common.gemini_client import upload, delete, generate_text, generate_image, extract_json
from artifact.pipeline.m1_fuse import load_bundle

ONESHOT = """Here are all evidence photographs and witness statements of one incident.

WITNESS STATEMENTS:
{testimonies}

Reconstruct what happened and return ONLY JSON:
{{"narrative": "<what happened>", "timeline": [{{"t":"T0","event":"..."}}],
 "keyframe_prompt": "<photographic description of the key moment, no blood/gore/injuries>",
 "motion_hint": "<one sentence of motion>"}}"""


def e2e(sid, rundir):
    captions, testimonies, images = load_bundle(sid)
    tst = "\n\n".join(f"--- {t['file']} ({t['witness']}) ---\n{t['text']}" for t in testimonies)
    files = [upload(p) for p in images]
    try:
        text, cost = generate_text(files + [ONESHOT.format(testimonies=tst)])
    finally:
        for f in files:
            delete(f)
    fusion = extract_json(text)
    fusion["gemini_rounds"] = 1

    key = Path(rundir) / "keyframe.png"
    ok, c2 = generate_image(fusion["keyframe_prompt"], key)
    cost += c2
    if not ok:
        from artifact.dataset.gen_evidence_images import cosmos_t2i
        cosmos_t2i(fusion["keyframe_prompt"], key)
        fusion["keyframe_backend"] = "cosmos"
    else:
        fusion["keyframe_backend"] = "gemini"
    fusion["cost_usd"] = round(cost, 4)
    json.dump(fusion, open(Path(rundir) / "fusion.json", "w"), indent=2, ensure_ascii=False)
    return fusion, cost
