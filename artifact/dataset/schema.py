"""scenario.json schema + validation."""

REQUIRED_TOP = ["id", "case_type", "split", "title", "ground_truth", "location",
                "participants", "evidence_images", "testimonies", "distractors"]
REQUIRED_GT = ["summary", "timeline", "key_physical_fact"]
EV_TYPES = {"victim", "perpetrator", "object", "scene"}

CASE_TYPES = {          # type -> (count, dev_count_within)
    "fall": 4, "traffic": 4, "burglary": 4, "altercation": 3,
    "arson": 3, "drowning": 3, "property": 4,
}
DEV_IDS = {"S04", "S08", "S12", "S15", "S25"}   # last of the 4/4/4/3/4 blocks -> 5 dev


def validate(s: dict) -> list:
    errs = []
    for k in REQUIRED_TOP:
        if k not in s:
            errs.append(f"missing top-level '{k}'")
    gt = s.get("ground_truth", {})
    for k in REQUIRED_GT:
        if k not in gt:
            errs.append(f"missing ground_truth.'{k}'")
    if not isinstance(gt.get("timeline"), list) or len(gt.get("timeline", [])) < 3:
        errs.append("ground_truth.timeline needs >=3 events")
    evs = s.get("evidence_images", [])
    if not 4 <= len(evs) <= 7:
        errs.append(f"need 4-7 evidence images, got {len(evs)}")
    types = {e.get("type") for e in evs}
    if "scene" not in types:
        errs.append("need at least one 'scene' evidence image")
    for e in evs:
        if e.get("type") not in EV_TYPES:
            errs.append(f"bad evidence type {e.get('type')}")
        for k in ("file", "desc", "caption"):
            if k not in e:
                errs.append(f"evidence image missing '{k}'")
    tst = s.get("testimonies", [])
    if not 2 <= len(tst) <= 3:
        errs.append(f"need 2-3 testimonies, got {len(tst)}")
    for t in tst:
        for k in ("file", "witness", "reliability", "text"):
            if k not in t:
                errs.append(f"testimony missing '{k}'")
    return errs
