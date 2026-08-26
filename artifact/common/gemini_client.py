"""Gemini access layer: text, vision, image generation, Veo video.

Skeleton lifted from agent_refine/agent_refine_queue.py (_retry, upload+ACTIVE poll).
Every call returns (result, cost_usd) so manifests can accumulate spend.
"""
import json, os, time
from pathlib import Path

from google import genai
from google.genai import types

from . import paths

_client = None
_client_lock = __import__("threading").Lock()


def client() -> genai.Client:
    global _client
    if _client is None:
        with _client_lock:                 # racing workers used to replace (and close) each other's client
            if _client is None:
                _client = genai.Client(api_key=paths.gemini_key(),
                                       http_options=types.HttpOptions(timeout=120_000))
    return _client


def _reset_client():
    global _client
    with _client_lock:
        _client = None


def _retry(fn, tries=6, base=10):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            msg = str(e)
            if "client has been closed" in msg:
                _reset_client()             # rebuild the shared client and retry
                time.sleep(2)
                continue
            if any(s in msg for s in ("503", "UNAVAILABLE", "500", "502", "504", "429", "overloaded",
                                      "RESOURCE_EXHAUSTED", "FAILED_PRECONDITION", "not in an ACTIVE state",
                                      "INTERNAL", "DeadlineExceeded",
                                      "403", "PERMISSION_DENIED")):  # files API throttling surfaces as 403 bursts
                wait = base * (i + 1)
                print(f"    gemini transient ({msg[:70]}); retry {i+1}/{tries} in {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
    raise last


def _cost(model, usage):
    p = paths.GEMINI_PRICE.get(model, {"in": 2.0, "out": 12.0})
    tin = getattr(usage, "prompt_token_count", 0) or 0
    tout = getattr(usage, "candidates_token_count", 0) or 0
    return (tin * p["in"] + tout * p["out"]) / 1e6


def upload(path):
    """Upload a media file and poll until ACTIVE (agent_refine's hardened loop)."""
    c = client()
    f = None
    for up in range(6):
        try:
            f = c.files.upload(file=str(path))
            for _ in range(120):
                if f.state.name in ("ACTIVE", "FAILED"):
                    break
                time.sleep(3)
                f = c.files.get(name=f.name)
            if f.state.name == "ACTIVE":
                return f
            reason = "processing FAILED"
        except Exception as e:
            reason = f"upload error ({str(e)[:70]})"
            f = None
        print(f"    gemini {reason} (attempt {up+1}/6); re-uploading {Path(path).name}", flush=True)
        try:
            if f is not None:
                c.files.delete(name=f.name)
        except Exception:
            pass
        time.sleep(8 * (up + 1))
    raise RuntimeError(f"gemini could not process {path} after 6 uploads")


def delete(f):
    try:
        client().files.delete(name=f.name)
    except Exception:
        pass


def generate_text(prompt_parts, model=None):
    """prompt_parts: list of str / uploaded files / PIL images. Returns (text, cost)."""
    model = model or paths.GEMINI_PRO
    r = _retry(lambda: client().models.generate_content(model=model, contents=prompt_parts))
    return r.text, _cost(model, r.usage_metadata)


def generate_image(prompt, out_path, reference_images=None, model=None):
    """Text (+ optional reference PIL images) -> PNG at out_path. Returns (ok, cost)."""
    model = model or paths.GEMINI_IMAGE
    contents = list(reference_images or []) + [prompt]

    def _go():
        r = client().models.generate_content(model=model, contents=contents)
        cost = _cost(model, r.usage_metadata) + paths.GEMINI_PRICE[model].get("per_image", 0.12)
        parts = (r.candidates and r.candidates[0].content
                 and r.candidates[0].content.parts) or []   # None => safety block
        for part in parts:
            data = getattr(part, "inline_data", None)
            if data and data.data:
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                Path(out_path).write_bytes(data.data)
                return True, cost
        # no image part -> treat as refusal, caller decides fallback
        txt = "".join(getattr(p, "text", "") or "" for p in parts)
        print(f"    image refused for {Path(out_path).name}: {txt[:120]}", flush=True)
        return False, cost

    return _retry(_go)


def generate_video_veo(prompt, image_path, out_path, model=None, timeout_s=1200):
    """Veo i2v: first-frame image + motion prompt -> mp4. Returns (ok, cost).
    Walks paths.VEO_CHAIN so a quota-exhausted model pool fails over quickly."""
    chain = [model] if model else paths.VEO_CHAIN
    c = client()
    img = types.Image.from_file(location=str(image_path)) if image_path else None
    op = None
    for i, m in enumerate(chain):
        try:
            op = _retry(lambda m=m: c.models.generate_videos(
                model=m, prompt=prompt, image=img,
                config=types.GenerateVideosConfig(aspect_ratio="16:9", resolution="720p")),
                tries=2 if i < len(chain) - 1 else 6)
            model = m
            break
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                print(f"    veo pool {m} exhausted; trying next", flush=True)
                continue
            raise
    if op is None:
        return False, 0.0
    t0 = time.time()
    while not op.done:
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"veo timeout for {out_path}")
        time.sleep(10)
        op = _retry(lambda: c.operations.get(op))
    if op.error or not getattr(op.response, "generated_videos", None):
        print(f"    veo failed for {Path(out_path).name}: {op.error}", flush=True)
        return False, 0.0
    vid = op.response.generated_videos[0]
    c.files.download(file=vid.video)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    vid.video.save(str(out_path))
    return True, 8 * paths.VEO_PRICE_PER_SECOND      # veo 3.1 clips are 8s


def inline_video(path, mime="video/mp4"):
    """Inline media part (bypasses the Files API, whose file-backed reads 403 under heavy use)."""
    return types.Part.from_bytes(data=Path(path).read_bytes(), mime_type=mime)


def extract_json(text):
    """Last fenced json block, else outermost braces (agent_refine's parser)."""
    if "```" in text:
        for b in reversed([b for b in text.split("```") if b.strip()]):
            bb = b.lstrip("json").strip()
            if bb.startswith("{") or bb.startswith("["):
                try:
                    return json.loads(bb)
                except Exception:
                    pass
    i, j = text.find("{"), text.rfind("}")
    if i == -1:
        i, j = text.find("["), text.rfind("]")
    return json.loads(text[i:j + 1])
