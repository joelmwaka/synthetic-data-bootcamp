"""VLM critic and slot-constrained prompt rewriter for night relighting."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from aieng.syn_data.image.generate.night import LOCKED_SLOTS, PROMPT_SLOTS
from aieng.syn_data.image.judge.vlm_api import vision_chat


CRITERIA = (
    "darkness_realism",
    "headlight_geometry",
    "falloff",
    "structure_preservation",
    "no_hallucination",
    "photorealism",
)

_SETUP = (
    "Physical setup: the camera sits 2.8 m above the rails inside the driver's cab, "
    "looking forward. The train's headlights are mounted on the train front about "
    "1 m above the rails - BELOW and slightly AHEAD of the camera - so the lamps "
    "themselves are never in frame. At night the brightest area should be the track "
    "and ballast in the lower centre of the frame, light should fall off quickly with "
    "distance along the track, lit rail heads may glint, lower parts of nearby objects "
    "are lit more than their tops, and the sky and anything outside the beam should be "
    "near-black (unless a light source exists in the scene)."
)

CRITIC_PROMPT = f"""You are a strict reviewer of synthetic training data for railway perception.
Image 1 is a real forward-facing train-cab photo taken at dusk/day.
Image 2 is a synthetic NIGHT version of the same frame that should keep every rail,
switch, signal, pole, building and tree in the same place.
{_SETUP}

Score image 2 from 0 to 10 on each criterion:
- darkness_realism: looks like a real camera frame at night (not just a darkened or blue-tinted day photo)
- headlight_geometry: the light pattern is consistent with lamps below the camera (pool on the near track, no visible lamp sources, no lens flare from above)
- falloff: brightness decreases plausibly with distance along the track
- structure_preservation: rails, switches, sleepers, signals and buildings keep the same layout and shape as image 1
- no_hallucination: nothing important is added, removed or moved
- photorealism: free of painterly, CGI or AI artifacts
- overall: would you keep this frame as night training data for a switch detector?

Respond with ONLY one JSON object, no markdown:
{{"darkness_realism": <0-10>, "headlight_geometry": <0-10>, "falloff": <0-10>,
"structure_preservation": <0-10>, "no_hallucination": <0-10>, "photorealism": <0-10>,
"overall": <0-10>, "critique": "<the single biggest physical or visual problem>",
"prompt_fix": "<one concrete instruction that would fix it>"}}"""


@dataclass
class NightCritique:
    """Parsed VLM critique; ``overall`` is ``None`` when parsing failed."""

    scores: dict[str, float]
    overall: float | None
    critique: str
    prompt_fix: str
    raw: str
    error: str | None = None


@dataclass
class RewriteResult:
    """Rewritten prompt slots plus what the rewriter changed."""

    slots: dict[str, str]
    rationale: str
    changed: list[str] = field(default_factory=list)
    raw: str = ""
    error: str | None = None


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object from model text; raise ``ValueError`` if absent."""
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if fence:
        raw = fence.group(1)
    else:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match is None:
            raise ValueError(f"No JSON object in response: {text[:160]!r}")
        raw = match.group(0)
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("JSON response is not an object")
    return data


def _score(value: Any) -> float:
    score = float(value)
    if not 0.0 <= score <= 10.0:
        raise ValueError(f"Score out of range: {score}")
    return score


def critique_night(
    original: Image.Image,
    candidate: Image.Image,
    *,
    model: str,
    api_base_url: str | None = None,
    provider: Any = None,
    api_key: str | None = None,
    max_side: int = 1024,
) -> NightCritique:
    """Score a night candidate against the real frame with a physics rubric."""
    text = vision_chat(
        CRITIC_PROMPT,
        [original, candidate],
        model=model,
        provider=provider,
        api_key=api_key,
        api_base_url=api_base_url,
        max_side=max_side,
    )
    try:
        data = parse_json_object(text)
        scores = {key: _score(data[key]) for key in CRITERIA}
        overall = _score(data["overall"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return NightCritique({}, None, "", "", raw=text, error=str(exc))
    return NightCritique(
        scores=scores,
        overall=overall,
        critique=str(data.get("critique", "")),
        prompt_fix=str(data.get("prompt_fix", "")),
        raw=text,
    )


REWRITE_PROMPT = """You are optimising an image-EDIT instruction for a small few-step
diffusion editor (FLUX.2 Klein) that relights a train-cab photo as night.
Image 1 is the real frame. Image 2 is the best result so far.
{setup}

The instruction is split into named slots (JSON below). Rewrite it to fix the
problems in the review history while keeping what already works.
Rules:
- Return the SAME keys. Do not change the locked slots: {locked}.
- Keep each slot to one or two short sentences (<= 45 words).
- Describe the desired result positively; the editor follows "the sky is uniformly
  dark" better than "no lights in the sky".
- Do not invent scene content that is not in image 1.

Current slots:
{slots}

Review history (oldest first):
{history}

Respond with ONLY one JSON object, no markdown:
{{"slots": {{<same keys>}}, "rationale": "<what you changed and why>"}}"""


def rewrite_prompt(
    slots: dict[str, str],
    history: list[dict[str, Any]],
    original: Image.Image,
    best_candidate: Image.Image,
    *,
    model: str,
    api_base_url: str | None = None,
    provider: Any = None,
    api_key: str | None = None,
    max_words: int = 60,
) -> RewriteResult:
    """Ask the VLM for a slot-wise rewrite; locked slots and keys are enforced.

    Any malformed response returns the input slots unchanged with ``error`` set.
    """
    prompt = REWRITE_PROMPT.format(
        setup=_SETUP,
        locked=", ".join(LOCKED_SLOTS),
        slots=json.dumps(slots, indent=2),
        history=json.dumps(history, indent=2),
    )
    text = vision_chat(
        prompt,
        [original, best_candidate],
        model=model,
        provider=provider,
        api_key=api_key,
        api_base_url=api_base_url,
    )
    try:
        data = parse_json_object(text)
        proposed = data["slots"]
        if not isinstance(proposed, dict) or set(proposed) != set(slots):
            raise ValueError(
                f"Rewriter returned keys {sorted(proposed) if isinstance(proposed, dict) else proposed!r}"
            )
        new_slots = {}
        for key in PROMPT_SLOTS:
            if key not in slots:
                continue
            value = slots[key] if key in LOCKED_SLOTS else str(proposed[key]).strip()
            if len(value.split()) > max_words:
                raise ValueError(f"Slot {key!r} exceeds {max_words} words")
            new_slots[key] = value
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return RewriteResult(dict(slots), "", raw=text, error=str(exc))
    changed = [key for key in new_slots if new_slots[key] != slots[key]]
    return RewriteResult(
        new_slots, str(data.get("rationale", "")), changed=changed, raw=text
    )
