"""Claude access via claude-agent-sdk (Claude Code auth path — no raw-API 429s on Max).

ask(task, model) -> (text, cost_usd). Image inputs: pass file paths; the agent reads
them with its Read tool (the trick from cosmos/upsampler/upsample.py).
"""
import asyncio

from claude_agent_sdk import query, ClaudeAgentOptions

from . import paths
from .gemini_client import extract_json  # re-export for callers


def ask(task, model=None, allow_read=False):
    model = model or paths.CLAUDE_OPUS

    async def _run():
        opts = ClaudeAgentOptions(model=model,
                                  allowed_tools=["Read"] if allow_read else [])
        parts, cost = [], 0.0
        async for msg in query(prompt=task, options=opts):
            for b in getattr(msg, "content", []) or []:
                t = getattr(b, "text", None)
                if t:
                    parts.append(t)
            if type(msg).__name__ == "ResultMessage":
                cost = getattr(msg, "total_cost_usd", 0.0) or 0.0
        return "".join(parts), cost

    return asyncio.run(_run())


def ask_json(task, model=None, allow_read=False, tries=3):
    """ask() + robust JSON extraction, retrying on parse failure."""
    last = None
    for i in range(tries):
        text, cost = ask(task, model=model, allow_read=allow_read)
        try:
            return extract_json(text), cost
        except Exception as e:
            last = e
            task = task + "\n\nYour previous reply was not parseable JSON. Reply with ONLY the JSON."
    raise RuntimeError(f"claude returned unparseable JSON after {tries} tries: {last}")
