"""Shared photographic-style and content-safety constants for all image prompts.

Every generated image must read as sober forensic documentation, never graphic content.
Both the Gemini and Cosmos t2i paths prepend/append these verbatim.
"""

STYLE_PREFIX = ("Forensic crime scene photography, official evidence documentation style, "
                "neutral clinical lighting, realistic, slightly desaturated colors, ")

SAFETY_SUFFIX = (". No blood, no gore, no visible injuries, no weapon pointed at any person, "
                 "all subjects fully clothed, respectful documentary framing, wide or medium shot.")

# per-evidence-type framing templates; {desc} comes from scenario.json
TYPE_TEMPLATES = {
    "victim": "medium shot of {desc}, photographed during scene examination after rescue, "
              "sitting or standing, uninjured appearance, identity-documentation framing",
    "perpetrator": "front-facing identification photograph of {desc}, plain background, "
                   "neutral expression, booking-photo framing",
    "object": "close-up of {desc} inside a labeled transparent evidence bag next to a forensic "
              "ruler scale on a clean table",
    "scene": "wide-angle view of {desc}, yellow numbered evidence markers placed on the ground, "
             "unoccupied scene, investigation-in-progress atmosphere",
}


def build_image_prompt(ev_type: str, desc: str) -> str:
    body = TYPE_TEMPLATES.get(ev_type, "{desc}").format(desc=desc)
    return STYLE_PREFIX + body + SAFETY_SUFFIX
