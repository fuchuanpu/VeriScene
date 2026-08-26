"""Module 1 — iterative multimodal evidence fusion.

Gemini drafts a structured account from ALL evidence (images + testimonies); the brain
LLM audits coverage/contradictions and issues directed follow-ups; Gemini revises.
Append-only (agent_refine style): the draft is revised, the follow-ups accumulate.

The pipeline NEVER sees ground_truth — only the evidence bundle.
"""
import json
from pathlib import Path

from artifact.common import paths
from artifact.common.gemini_client import upload, delete, generate_text, generate_image, extract_json
from artifact.common.claude_client import ask_json as claude_json

FUSE = """You are a forensic evidence-fusion analyst. You are given the complete evidence of one case:
photographs (victim/perpetrator identification, bagged physical evidence, scene examination shots with
markers) and written witness statements. Some statements may be partially unreliable or contradict others.

EVIDENCE CAPTIONS:
{captions}

WITNESS STATEMENTS:
{testimonies}

DOCUMENTATION PROTOCOL (applies to all cases): person photographs are taken during scene examination
AFTER any rescue/medical response, so victims appear recovered and uninjured by protocol — an uninjured
appearance in a photo is NOT evidence that the incident did not occur or was staged. Infer what happened
from the physical scene evidence and testimony.

Fuse ALL evidence into one coherent reconstruction of what physically happened. Weigh testimony against
physical evidence; physical evidence wins conflicts. Cite evidence for every claim.

{followups}

Write a brief analysis, then exactly one fenced ```json block:
{{
 "narrative": "<6-10 sentence account of what happened, each claim traceable to evidence>",
 "timeline": [{{"t":"T0","event":"...","evidence_refs":["scene_1.png","w2.txt"]}}, ... 4-6 steps],
 "contradictions": [{{"between":["w1.txt","w3.txt"],"issue":"...","resolution":"..."}}],
 "keyframe_prompt": "<one detailed photographic description of the single moment that best captures the
   incident's key physical event, consistent with the scene photos: camera position, lighting, subjects,
   positions/poses, environment. No blood, no gore, no visible injuries.>",
 "motion_hint": "<one sentence: the COMPLETE physical event from that moment through its outcome —
   e.g. for a fall: the support fails, the person drops, and comes to rest — never a truncated version>"
}}"""

AUDIT = """You are auditing a draft forensic reconstruction against the evidence inventory.

EVIDENCE INVENTORY (files + captions):
{captions}
WITNESS FILES: {witness_files}

DRAFT (JSON):
{draft}

Check: (1) every evidence file is either used in some evidence_refs or explicitly irrelevant;
(2) every contradiction between statements is surfaced and resolved by physical evidence;
(3) the timeline is physically ordered (causes before effects, no impossible jumps).

Return ONLY JSON: {{"complete": <bool>, "followups": ["<directed question or instruction for the analyst,
one per gap, max 3>"]}}. complete=true with empty followups if the draft holds."""


def brain_json(task, brain):
    b = paths.BRAINS[brain]
    if b["kind"] == "claude":
        return claude_json(task, model=b["model"])
    text, cost = generate_text([task], model=b["model"])
    return extract_json(text), cost


def load_bundle(sid):
    """Evidence-only view of a scenario (ground_truth stripped)."""
    s = json.load(open(paths.scenario_dir(sid) / "scenario.json"))
    d = paths.scenario_dir(sid)
    captions = [{"file": e["file"], "type": e["type"], "caption": e["caption"]}
                for e in s["evidence_images"]]
    testimonies = [{"file": t["file"], "witness": t["witness"],
                    "text": (d / "testimony" / t["file"]).read_text()}
                   for t in s["testimonies"]]
    images = [d / "evidence" / e["file"] for e in s["evidence_images"]
              if (d / "evidence" / e["file"]).exists()]
    return captions, testimonies, images


def fuse(sid, rundir, brain="opus", rounds=2, single_shot=False):
    """Returns (fusion_dict, cost). single_shot=True is the w/o-fusion ablation."""
    captions, testimonies, images = load_bundle(sid)
    cap_txt = json.dumps(captions, indent=1)
    tst_txt = "\n\n".join(f"--- {t['file']} ({t['witness']}) ---\n{t['text']}" for t in testimonies)

    from PIL import Image
    files = [Image.open(p) for p in images]   # inline, not Files API (403s under heavy use)
    total = 0.0
    if True:
        followups = []
        fusion = None
        for rnd in range(1 if single_shot else rounds + 1):
            fu = ""
            if followups:
                fu = ("AUDITOR FOLLOW-UPS from the previous round — address each explicitly, "
                      "revising the draft (do not discard prior correct content):\n- " + "\n- ".join(followups))
            prompt = FUSE.format(captions=cap_txt, testimonies=tst_txt, followups=fu)
            gmodel = paths.BRAINS[brain].get("gemini", paths.GEMINI_PRO)
            text, c = generate_text(files + [prompt], model=gmodel)
            total += c
            fusion = extract_json(text)
            if single_shot or rnd == rounds:
                break
            audit, c2 = brain_json(
                AUDIT.format(captions=cap_txt, draft=json.dumps(fusion, indent=1)[:6000],
                             witness_files=[t["file"] for t in testimonies]), brain)
            total += c2
            new = [q for q in audit.get("followups", []) if q not in followups][:3]
            print(f"    m1 round {rnd}: complete={audit.get('complete')} +{len(new)} followups", flush=True)
            if audit.get("complete") or not new:
                break
            followups.extend(new)
        fusion["gemini_rounds"] = rnd + 1

    # keyframe: POLICY (2026-08-15) — local Cosmos t2i by default; Gemini image only
    # when ALLOW_GEMINI_IMAGE=1 (paid path, avoid unless necessary).
    import os
    key = Path(rundir) / "keyframe.png"
    ok = False
    if os.environ.get("ALLOW_GEMINI_IMAGE") == "1":
        from PIL import Image
        refs = [Image.open(p) for p in images[:3]]
        ok, c3 = generate_image(
            fusion["keyframe_prompt"] + " Photorealistic, consistent with the reference scene photographs.",
            key, reference_images=refs)
        total += c3
    if ok:
        fusion["keyframe_backend"] = "gemini"
    else:
        from artifact.dataset.gen_evidence_images import cosmos_t2i
        cosmos_t2i(fusion["keyframe_prompt"], key)
        fusion["keyframe_backend"] = "cosmos"

    fusion["cost_usd"] = round(total, 4)
    json.dump(fusion, open(Path(rundir) / "fusion.json", "w"), indent=2, ensure_ascii=False)
    return fusion, total
