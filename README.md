# VeriScene

VeriScene is an agent that **orchestrates a video world model** to reconstruct
incident scenes from multimodal legal evidence — forensic photographs and
witness statements of varying reliability — keeping every claim traceable to
evidence and every rendered motion physically plausible.

The pipeline chains three generation–verification loops:

1. **Iterative Evidence Fusion** (`artifact/pipeline/m1_fuse.py`) — a VLM fuses
   the evidence inventory into a cited draft narrative; an auditor agent flags
   uncovered items and unresolved contradictions and issues targeted
   follow-ups (bounded rounds, append-only).
2. **World-Model Physical Reasoning** (`artifact/pipeline/m2_reason.py`) — the
   hypothesized key event is rolled out as a short low-resolution probe clip in
   a world foundation model; a physics critic detects violations and injects
   corrective constraints.
3. **Re-enactment Rendering** (`artifact/pipeline/m3_render.py`) — the audited
   narrative and validated constraints compile into a physics-annotated prompt
   rendered as the final re-enactment video (hosted renderer with a local
   world-model fallback).

## Benchmark

The 25-scenario multimodal legal-evidence benchmark (scenario ground truths,
139 forensic-style photographs, 61 witness statements, and per-run scores)
will be released in a subsequent update of this repository; this drop contains
the complete pipeline, baseline, and evaluation code.

> **Anonymity notice (double-blind submission).** All persons, locations, and
> incidents in this benchmark are fictional and synthetically generated; the
> artifact contains no real case data and no personally identifying
> information.

## Quickstart

```bash
pip install -r requirements.txt
cp artifact/scripts/env.sh.example artifact/scripts/env.sh   # fill in your keys
source artifact/scripts/env.sh

# reproduce one scene end-to-end once the benchmark data is in data/scenarios
# (needs a local world-model server, see below)
python -m artifact.pipeline.run_scene --scene S01 --method ours
python -m artifact.eval.judge --only S01 && python -m artifact.eval.collect
```

The local world-model fallback expects an NVIDIA Cosmos-family pipeline served
via the file-queue protocol in `artifact/common/nano_queue.py`
(`artifact/scripts/start_nano_server.sh` shows the reference launcher; one
80GB-class GPU suffices at the reduced probe resolution).

## Layout

```
artifact/   pipeline (3 modules) · baselines · dataset construction · LLM-judge eval
```
