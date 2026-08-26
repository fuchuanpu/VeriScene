"""Module 2 — world-model physical reasoning.

The brain turns the fused timeline into a physics-annotated motion prompt; optionally a
Cosmos3-Nano short probe clip is generated and critiqued (agent_refine's violation-ranking
critic) and the constraints are refined once. Skipped entirely by the w/o-reasoning ablation.
"""
import json
from pathlib import Path

from artifact.common import paths
from artifact.common.gemini_client import upload, delete, generate_text
from artifact.common.nano_queue import queue_generate
from artifact.pipeline.m1_fuse import brain_json

TO_PHYSICS = """You are a physical-dynamics expert preparing a video world-model prompt from a forensic
reconstruction. The clip starts from a fixed keyframe image; the prompt drives the motion.

RECONSTRUCTION:
{fusion}

Produce the motion specification for an 8-second clip of the KEY physical event, covering the event
from onset through its physical outcome (never truncated midway), obeying real physics
(gravity, momentum, contact forces, continuous trajectories, mechanism-causes-effect). Non-graphic:
no blood, no gore, no injury detail.

Return ONLY JSON:
{{
 "video_prompt": "<4-7 sentence motion description: who/what moves, direction, speed, cause->effect order>",
 "physics_constraints": ["<explicit physical rule the clip must respect>", ... 3-5],
 "negative_terms": ["<failure mode to suppress: teleporting, floating, frozen scene...>", ... 3-5]
}}"""

PROBE_CRITIC = """This is a short AI-generated probe clip of a reconstructed incident. The intended motion:
"{motion}"
Check ONLY physics-law violations (gravity, causality/mechanism, momentum, trajectory continuity, object
permanence, impenetrability, support/contact) — ignore rendering artifacts (morphing, blur, flicker).
Return ONLY JSON: {{"physical_plausible": <bool>, "violations": [{{"what":"","law":"","severity":
"low|moderate|severe"}}], "fix_constraints": ["<at most 2 added rules fixing the worst violation>"],
"fix_negative": ["<at most 2 terms>"]}}"""


def reason(sid, rundir, fusion, brain="opus", probe=True):
    rundir = Path(rundir)
    spec, total = brain_json(TO_PHYSICS.format(fusion=json.dumps(
        {k: fusion[k] for k in ("narrative", "timeline", "motion_hint")}, indent=1)), brain)

    if probe:
        pos = rundir / "probe_pos.json"
        json.dump({"scene_description": spec["video_prompt"],
                   "physics_constraints": spec["physics_constraints"]}, open(pos, "w"))
        neg = rundir / "probe_neg.json"
        json.dump({"violation_blacklist": spec["negative_terms"]}, open(neg, "w"))
        clip = rundir / "probe.mp4"
        try:
            queue_generate(pos, clip, neg_json_path=neg, image=rundir / "keyframe.png",
                           num_frames=57, steps=25, timeout_s=1200,
                           height=480, width=832)   # low-res probe: survives GPU contention
            from artifact.common.gemini_client import inline_video
            text, c = generate_text([inline_video(clip),
                                     PROBE_CRITIC.format(motion=fusion["motion_hint"])],
                                    model=paths.BRAINS[brain].get("gemini", paths.GEMINI_PRO))
            total += c
            from artifact.common.gemini_client import extract_json
            verdict = extract_json(text)
            spec["probe_verdict"] = verdict
            spec["physics_constraints"] += [x for x in verdict.get("fix_constraints", [])[:2]
                                            if x not in spec["physics_constraints"]]
            spec["negative_terms"] += [x for x in verdict.get("fix_negative", [])[:2]
                                       if x not in spec["negative_terms"]]
            print(f"    m2 probe: plausible={verdict.get('physical_plausible')} "
                  f"+{len(verdict.get('fix_constraints', []))} constraints", flush=True)
        except Exception as e:
            print(f"    m2 probe skipped ({str(e)[:80]})", flush=True)
            spec["probe_verdict"] = {"error": str(e)[:200]}

    spec["cost_usd"] = round(total, 4)
    json.dump(spec, open(rundir / "reasoning.json", "w"), indent=2, ensure_ascii=False)
    return spec, total
