"""Tests for the generator registry (no GPU, no network)."""

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from aieng.syn_data.image.generate import vlm_generate
from aieng.syn_data.image.generate.compare_methods import ALL_COMPARE_METHODS
from aieng.syn_data.image.workflow import generators as gens
from aieng.syn_data.image.workflow import night as wf


BASE = OmegaConf.create(
    {
        "family": "sd15",
        "base_model_id": gens.SD15,
        "inpaint_model_id": gens.KLEIN_4B,
        "instruct_model_id": gens.KLEIN_4B,
        "instruct_num_inference_steps": 4,
        "instruct_guidance_scale": 1.0,
        "controlnet": {
            "depth": "lllyasviel/sd-controlnet-depth",
            "seg": "lllyasviel/sd-controlnet-seg",
        },
        "controlnet_scale": {"depth": 0.75, "seg": 0.45},
        "vlm_api_enabled": False,
        "seed": 42,
    },
)


def test_registry_is_consistent():
    assert gens.DEFAULT_GENERATOR in gens.GENERATORS
    presets = set(wf.PRESET_DESCRIPTIONS)
    for spec in gens.GENERATORS.values():
        assert spec.method in ALL_COMPARE_METHODS
        assert spec.recommended_preset in presets
        assert len({p.name for p in spec.params}) == len(spec.params)
        for param in spec.params:
            assert param.min <= param.default <= param.max
        if spec.max_prompt_tokens:
            assert spec.recommended_preset in {"caption", "short_instruction"}
        assert spec.cloud == (not spec.repos)
    with pytest.raises(KeyError, match="choose from"):
        gens.get_generator("nope")


def test_effective_cfg_applies_overrides_and_nested_params():
    spec = gens.get_generator("sd15_controlnet")
    cfg = gens.effective_generation_cfg(
        BASE, spec, {"strength": 0.6, "depth_scale": 1.1, "steps": 25.4}
    )
    assert cfg.controlnet_strength == 0.6
    assert cfg.controlnet_scale.depth == 1.1 and cfg.controlnet_scale.seg == 0.5
    assert cfg.num_inference_steps == 25
    assert cfg.controlnet_strength_cap_global == 0.95
    assert "night_headlights" in cfg.global_anomaly_ids
    assert BASE.controlnet_scale.depth == 0.75  # input untouched
    pix = gens.effective_generation_cfg(BASE, gens.get_generator("instruct_pix2pix"))
    assert (
        pix.instruct_model_id == gens.INSTRUCT_PIX2PIX
        and pix.instruct_num_inference_steps == 30
    )
    assert pix.instruct_guidance_scale == 7.5
    with pytest.raises(ValueError, match="unknown params"):
        gens.effective_generation_cfg(BASE, spec, {"bogus": 1})


def test_params_are_clamped_and_rounded():
    spec = gens.get_generator("klein4b_img2img")
    assert spec.resolve_params({"strength": 5, "steps": 2.6}) == {
        "strength": 1.0,
        "steps": 3,
    }
    assert spec.resolve_params(None) == spec.default_params()


def test_comparer_key_ignores_params_but_tracks_models():
    instruct = gens.comparer_key(gens.get_generator("klein4b_instruct"), BASE)
    img2img = gens.comparer_key(gens.get_generator("klein4b_img2img"), BASE)
    pix = gens.comparer_key(gens.get_generator("instruct_pix2pix"), BASE)
    assert instruct != pix and instruct[2] == img2img[2] == gens.KLEIN_4B
    assert instruct == gens.comparer_key(gens.get_generator("klein4b_instruct"), BASE)


def test_availability_rules():
    def cached(_repo):
        return True

    def missing(_repo):
        return False

    klein = gens.get_generator("klein4b_instruct")
    assert (
        gens.availability(
            klein, api_key_available=False, vram_gb=22, is_cached=cached
        ).status
        == "ready"
    )
    dl = gens.availability(
        klein, api_key_available=False, vram_gb=22, is_cached=missing, offline=False
    )
    assert dl.status == "download" and dl.usable
    off = gens.availability(
        klein, api_key_available=False, vram_gb=22, is_cached=missing, offline=True
    )
    assert off.status == "blocked" and "HF_HUB_OFFLINE" in off.reason
    qwen = gens.get_generator("qwen_image_edit")
    big = gens.availability(qwen, api_key_available=True, vram_gb=22, is_cached=cached)
    assert big.status == "blocked" and "22 GB" in big.reason
    assert (
        gens.availability(
            qwen, api_key_available=True, vram_gb=80, is_cached=cached
        ).status
        == "ready"
    )
    api = gens.get_generator("nano_banana_api")
    assert not gens.availability(api, api_key_available=False).usable
    assert gens.availability(api, api_key_available=True).status == "ready"


def test_ade20k_palette_matches_segformer_ids():
    assert gens.ADE20K_PALETTE.shape == (150, 3)
    labels = np.array([[0, 2], [4, 6]])
    colours = gens.ade20k_colour_map(labels)
    assert colours[0, 0].tolist() == [120, 120, 120]  # wall
    assert colours[0, 1].tolist() == [6, 230, 230]  # sky
    assert colours[1, 0].tolist() == [4, 200, 3]  # tree
    assert colours[1, 1].tolist() == [140, 140, 140]  # road
    assert gens.ade20k_colour_map(np.array([[-1, 200]])).sum() == 0


def test_conditioning_from_maps():
    disparity = np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4)
    labels = np.full((3, 4), 2)
    depth, seg = gens.conditioning_from_maps(disparity, labels)
    assert depth.depth_map.shape == (3, 4) and seg.label_map.dtype == np.int32
    assert seg.colored_map[0, 0].tolist() == [6, 230, 230]
    with pytest.raises(ValueError, match="differ"):
        gens.conditioning_from_maps(disparity, labels[:2])


class _FakeComparer:
    def __init__(self, size=(96, 54), error=None):
        self.calls = []
        self.size = size
        self.error = error

    def run_method(self, method, image, **kwargs):
        self.calls.append((method, kwargs))
        return SimpleNamespace(image=Image.new("RGB", self.size), error=self.error)


def test_run_generator_passes_method_mask_negative_and_conditioning():
    comparer = _FakeComparer(size=(100, 50))
    src = Image.new("RGB", (96, 54))
    conditioning = gens.conditioning_from_maps(
        np.zeros((54, 96), np.float32), np.zeros((54, 96), int)
    )
    out = gens.run_generator(
        comparer,
        BASE,
        gens.get_generator("sd15_inpaint"),
        src,
        "night",
        params={"strength": 0.5},
        negative_prompt="day",
        conditioning=conditioning,
        work_size=(96, 54),
        seed_offset=3,
    )
    assert out.size == (96, 54)
    method, kwargs = comparer.calls[0]
    assert method == "inpaint" and kwargs["seed_offset"] == 3
    assert kwargs["anomaly_cfg"].edit_mask.mode == "full"
    assert kwargs["anomaly_cfg"].negative_prompt == "day"
    assert kwargs["generation_cfg"].inpaint_model_id == gens.SD15_INPAINT
    assert kwargs["generation_cfg"].strength == 0.5
    assert kwargs["depth"] is conditioning[0]

    gens.run_generator(
        comparer,
        BASE,
        gens.get_generator("klein4b_instruct"),
        src,
        "night",
        negative_prompt="ignored",
        work_size=(96, 54),
    )
    method, kwargs = comparer.calls[1]
    assert method == "instruct" and "negative_prompt" not in kwargs["anomaly_cfg"]
    assert kwargs["depth"] is None

    with pytest.raises(ValueError, match="conditioning"):
        gens.run_generator(
            comparer,
            BASE,
            gens.get_generator("sd15_controlnet"),
            src,
            "x",
            work_size=(96, 54),
        )
    with pytest.raises(RuntimeError, match="failed"):
        gens.run_generator(
            _FakeComparer(error="boom"),
            BASE,
            gens.get_generator("klein4b_instruct"),
            src,
            "x",
            work_size=(96, 54),
        )


def test_clip_token_count_flags_long_prompts():
    facts = {
        "environment": ["rural"],
        "weather": [],
        "track_layout": [],
        "n_tracks": 2,
        "n_switches": 1,
        "lamp_height_m": 1.0,
        "beam_reach_m": 40,
    }
    presets = wf.prompt_presets(facts)
    assert gens.clip_token_count(presets["caption"]["prompt"]) <= gens.CLIP_MAX_TOKENS
    assert (
        gens.clip_token_count(presets["short_instruction"]["prompt"])
        <= gens.CLIP_MAX_TOKENS
    )
    assert (
        gens.clip_token_count(presets["hybrid_prior"]["prompt"]) > gens.CLIP_MAX_TOKENS
    )


def test_vlm_edit_preamble_and_proxy_aspect_ratio(monkeypatch):
    default = vlm_generate._edit_prompt("darken")
    assert default.startswith("Edit this street photograph") and default.endswith(
        "Instruction: darken"
    )
    assert (
        vlm_generate._edit_prompt("darken", "Relight it.")
        == "Relight it. Instruction: darken"
    )

    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[])

    def _images_edit(**kwargs):
        captured["images_prompt"] = kwargs["prompt"]
        return SimpleNamespace(data=[SimpleNamespace(b64_json=None, url=None)])

    class _Client:
        chat = SimpleNamespace(completions=_Completions())
        images = SimpleNamespace(edit=_images_edit)

    monkeypatch.setattr(vlm_generate, "make_openai_client", lambda **k: _Client())
    cfg = vlm_generate.VlmGenerateConfig(
        api_base_url="http://proxy", aspect_ratio="16:9", edit_preamble="Relight."
    )
    with pytest.raises(RuntimeError):
        vlm_generate.generate_with_vlm(
            "night", seed_image=Image.new("RGB", (16, 9)), config=cfg
        )
    assert captured["extra_body"]["image_config"] == {"aspect_ratio": "16:9"}
    assert captured["messages"][0]["content"][0]["text"].startswith("Relight.")
    assert captured["images_prompt"].startswith("Relight.")


def test_candidate_records_default_to_klein_for_old_runs():
    old = wf.Candidate.from_dict({"label": "x", "input": "prior", "prompt": "p"})
    assert (
        old.generator == "klein4b_instruct"
        and old.generator_params == {}
        and old.negative_prompt == ""
    )
