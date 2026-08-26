"""Generate the 25 VeriScene scenarios (scenario.json + testimony txt files) with Claude.

One Claude call per case type (7 calls), each returning all scenarios of that type.
Usage:  python -m artifact.dataset.gen_scenarios [--regen S07]
"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor

from artifact.common import paths
from artifact.common.claude_client import ask_json
from artifact.dataset import schema

TYPE_BRIEFS = {
    "fall": "falls from height or staircases where physical evidence distinguishes accident from a push "
            "(impact position, railing damage, trajectory)",
    "traffic": "road traffic incidents reconstructed from skid marks, vehicle deformation, debris fields "
               "(collisions, hit-and-run, failed braking)",
    "burglary": "break-in path reconstruction from glass fracture direction, tool marks, footprint trails",
    "altercation": "non-graphic physical altercations evidenced by overturned furniture, dropped items, "
                   "bruise-free scene disturbance; any weapon appears only bagged as evidence",
    "arson": "fire-origin investigation from burn patterns, V-shaped scorch marks, accelerant containers",
    "drowning": "water incidents at docks, pools, riverbanks evidenced by railing damage, dropped belongings, "
                "wet-surface slip marks",
    "property": "property damage disputes: vehicle scrapes, broken fences, fallen trees onto cars, "
                "shattered storefront glass",
}

PROMPT = """You are building a synthetic benchmark of criminal/civil incident scenes for a research paper on
multimodal legal evidence fusion. Generate {n} DISTINCT scenarios of case type "{ctype}": {brief}.

Hard requirements for EVERY scenario:
- Physical-mechanism-driven: the ground truth hinges on a reconstructable physical event chain.
- Non-graphic: no blood, no gore, no injuries described visually, no weapon aimed at anyone.
- IDs: {ids} (in order). All file names lowercase snake_case.
- 4-7 evidence_images per scenario covering types: "victim" (person photographed unharmed during scene
  examination; omit for victimless property cases), "perpetrator" (id-photo style; omit if unknown offender
  is part of the ground truth), "object" (bagged physical evidence with ruler), "scene" (2-3 wide shots
  with evidence markers). Each: {{"file":"xx.png","type":"...","desc":"<visual description for an image
  generator, concrete and photographable>","caption":"<one-line forensic log caption>"}}.
- 2-3 testimonies per scenario: exactly one fully reliable, one partially unreliable (wrong time or
  position estimate), optionally one contradicting another witness. Each: {{"file":"w1.txt","witness":
  "<role, e.g. neighbor>","reliability":"reliable|partial|contradictory","discrepancy":"<what is off,
  or null>","text":"<150-250 word first-person statement in plain English, police-record register>"}}.
- ground_truth: {{"summary":"<5-8 sentence account of what actually happened>","timeline":[{{"t":"T0",
  "event":"..."}},... 4-6 steps],"key_physical_fact":"<the single physical inference that resolves the case>"}}
- 1-2 distractors: descriptions of irrelevant evidence present at the scene.
- Vary: indoor/outdoor, day/night, urban/rural, participant ages/genders/builds. All persons fictional.

Return ONLY a JSON array of {n} scenario objects, each:
{{"id":"...","case_type":"{ctype}","split":"main","title":"<short title>","ground_truth":{{...}},
"location":"...","participants":[{{"role":"victim|perpetrator|witness","desc":"..."}}],
"evidence_images":[...],"testimonies":[...],"distractors":[...]}}"""


def type_ids():
    """S01..S25 assigned in schema.CASE_TYPES order."""
    out, n = {}, 1
    for ctype, cnt in schema.CASE_TYPES.items():
        out[ctype] = [f"S{n+i:02d}" for i in range(cnt)]
        n += cnt
    return out


def gen_type(ctype, ids):
    task = PROMPT.format(n=len(ids), ctype=ctype, brief=TYPE_BRIEFS[ctype], ids=", ".join(ids))
    arr, cost = ask_json(task, model=paths.CLAUDE_OPUS)
    if isinstance(arr, dict):
        arr = arr.get("scenarios", [arr])
    print(f"  [{ctype}] {len(arr)} scenarios, ${cost:.2f}", flush=True)
    return arr, cost


def write_scenario(s):
    s["split"] = "dev" if s["id"] in schema.DEV_IDS else "main"
    errs = schema.validate(s)
    if errs:
        print(f"  !! {s.get('id','?')}: {errs}", flush=True)
        return False
    d = paths.scenario_dir(s["id"])
    (d / "testimony").mkdir(parents=True, exist_ok=True)
    (d / "evidence").mkdir(parents=True, exist_ok=True)
    for t in s["testimonies"]:
        (d / "testimony" / t["file"]).write_text(t["text"])
    json.dump(s, open(d / "scenario.json", "w"), indent=2, ensure_ascii=False)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", help="regenerate a single scenario id (regenerates its whole type batch "
                                    "but only overwrites this id)")
    args = ap.parse_args()
    tids = type_ids()
    todo = tids
    if args.regen:
        todo = {c: ids for c, ids in tids.items() if args.regen in ids}
        if not todo:
            sys.exit(f"unknown id {args.regen}")

    total_cost, ok = 0.0, 0
    with ThreadPoolExecutor(max_workers=7) as ex:
        futs = {ex.submit(gen_type, c, ids): c for c, ids in todo.items()}
        for f in futs:
            arr, cost = f.result()
            total_cost += cost
            for s in arr:
                if args.regen and s.get("id") != args.regen:
                    continue
                ok += write_scenario(s)
    print(f"DONE: {ok} scenarios written, total ${total_cost:.2f}", flush=True)


if __name__ == "__main__":
    main()
