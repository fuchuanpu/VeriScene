"""Central path/model constants for the VeriScene artifact."""
import os
from pathlib import Path

LAW = Path(__file__).resolve().parents[2]          # artifact root
WORLD = LAW.parent
DATA = LAW / "data"
SCENARIOS = DATA / "scenarios"
RESULTS = DATA / "results"
NANO_QUEUE = Path(os.environ.get("VERISCENE_NANO_QUEUE", WORLD / "nano_queue"))


for d in (SCENARIOS, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

# --- models -----------------------------------------------------------------
GEMINI_PRO = "gemini-3.1-pro-preview"              # fusion / critic / judge
GEMINI_FLASH = "gemini-3.5-flash"                  # cheap brain tier (verified name)
GEMINI_IMAGE = "gemini-3-pro-image-preview"        # evidence images / keyframes (verified)
VEO = "veo-3.1-fast-generate-preview"              # primary video backend
VEO_CHAIN = [VEO, "veo-3.1-lite-generate-preview",  # per-model quota pools: fall through
             "veo-3.1-generate-preview"]            # on RESOURCE_EXHAUSTED
CLAUDE_OPUS = "claude-opus-4-8"                    # same tier agent_refine used
CLAUDE_SONNET = "claude-sonnet-5"

# Gemini pricing per 1M tokens (USD) for cost accounting; images billed per image.
GEMINI_PRICE = {
    "gemini-3.1-pro-preview": {"in": 2.0, "out": 12.0},
    "gemini-3.5-flash": {"in": 0.30, "out": 2.50},
    "gemini-3-pro-image-preview": {"in": 2.0, "out": 0.0, "per_image": 0.12},
}
VEO_PRICE_PER_SECOND = 0.15                        # veo-3.1-fast, 720p

BRAINS = {                                          # --brain tiers for the LLM study
    # "gemini" = model used for the Gemini-side calls (fusion drafter / probe critic)
    "opus": {"kind": "claude", "model": CLAUDE_OPUS, "gemini": GEMINI_PRO},
    "sonnet": {"kind": "claude", "model": CLAUDE_SONNET, "gemini": GEMINI_PRO},
    "gemini-pro": {"kind": "gemini", "model": GEMINI_PRO, "gemini": GEMINI_PRO},
    "gemini-flash": {"kind": "gemini", "model": GEMINI_FLASH, "gemini": GEMINI_FLASH},
    "haiku": {"kind": "claude", "model": "claude-haiku-4-5-20251001", "gemini": GEMINI_FLASH},
}


def scenario_dir(sid: str) -> Path:
    return SCENARIOS / sid


def run_dir(sid: str, method: str, ts: str) -> Path:
    d = scenario_dir(sid) / "runs" / method / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("set GEMINI_API_KEY (see artifact/scripts/env.sh)")
    return key
