"""Model-holding engine shared by the Streamlit app and the batch worker.

GPU work (depth/segmentation on a cache miss, Klein edits) runs under a
cross-process file lock in the studio folder, so the app and a worker never
run models on the GPU at the same time.
"""

from __future__ import annotations

import gc
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aieng.syn_data.image.config import load_config
from aieng.syn_data.image.data.frames import Frame, FrameSource, frame_source_for
from aieng.syn_data.image.workflow import generators as gens
from aieng.syn_data.image.workflow import night as wf
from aieng.syn_data.image.workflow.store import RunStore, studio_root
from filelock import FileLock, Timeout
from omegaconf import DictConfig
from PIL import Image


_CONFIG_LOCK = threading.Lock()
API_KEY_ENV = ("OPENAI_API_KEY", "VECTOR_PROXY_API_KEY", "JUDGE_API_KEY")


class GpuBusyError(RuntimeError):
    """Raised when another process holds the GPU lock."""


def load_cfg(dataset: str, hardware: str, *, start: Path | None = None) -> DictConfig:
    """Compose the Hydra config (Hydra's global state is not thread-safe)."""
    with _CONFIG_LOCK:
        return load_config(start=start, overrides=[f"dataset_name={dataset}", f"hardware={hardware}"])


def vlm_key_available() -> bool:
    """Whether a judge API key is configured (never reveals the value)."""
    return any(os.environ.get(name) for name in API_KEY_ENV)


def free_cuda() -> None:
    """Release cached CUDA memory."""
    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - torch is a hard dependency of the image stack
        pass


ProgressFn = Callable[[str], None]


class Engine:
    """Config, data, storage, and lazily-loaded models for one dataset/hardware."""

    def __init__(self, cfg: DictConfig, *, gpu_timeout: float = 0.0):
        self.cfg = cfg
        self.root = studio_root(cfg)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = RunStore(self.root)
        self.cache = wf.GeometryCache(self.root / "cache")
        self.gpu_lock = FileLock(str(self.root / "gpu.lock"))
        self.gpu_timeout = gpu_timeout
        self._source: FrameSource | None = None
        self._comparer: Any = None
        self._comparer_key: tuple[str, ...] | None = None
        self._comparer_generator: str | None = None
        self._model_lock = threading.Lock()

    # --- data ----------------------------------------------------------------

    @property
    def source(self) -> FrameSource:
        """Frame source for the configured dataset."""
        if self._source is None:
            self._source = frame_source_for(self.cfg)
        return self._source

    def frame(self, frame_id: str) -> Frame:
        """Load one frame."""
        return self.source.get(frame_id)

    # --- models --------------------------------------------------------------

    def _acquire_gpu(self) -> FileLock:
        try:
            self.gpu_lock.acquire(timeout=self.gpu_timeout)
        except Timeout as exc:
            raise GpuBusyError("The GPU is busy (a batch worker or another session is generating).") from exc
        return self.gpu_lock

    @property
    def comparer_loaded(self) -> bool:
        """Whether a generator is loaded in this process."""
        return self._comparer is not None

    @property
    def loaded_generator(self) -> str | None:
        """Id of the generator whose models are loaded (``None`` if none)."""
        return self._comparer_generator if self._comparer is not None else None

    def _release_comparer(self) -> None:
        if self._comparer is not None:
            self._comparer.unload()
        self._comparer = None
        self._comparer_key = None
        self._comparer_generator = None
        free_cuda()

    def _build_comparer(self, spec: gens.GeneratorSpec) -> Any:
        from aieng.syn_data.image.compare_methods import MethodComparer  # noqa: PLC0415

        cfg = self.cfg.copy()
        cfg.generation = gens.effective_generation_cfg(self.cfg.generation, spec)
        return MethodComparer.from_config(cfg)

    def comparer(self, spec: gens.GeneratorSpec | None = None) -> Any:
        """``MethodComparer`` for ``spec``; switching models unloads the previous ones.

        Pipelines load lazily on the first edit. Generators that share models
        (e.g. Klein instruct and Klein img2img) reuse the same comparer.
        """
        spec = spec or gens.get_generator(gens.DEFAULT_GENERATOR)
        key = gens.comparer_key(spec, self.cfg.generation)
        if self._comparer is not None and key != self._comparer_key:
            self._release_comparer()
        if self._comparer is None:
            self._comparer = self._build_comparer(spec)
            self._comparer_key = key
        self._comparer_generator = spec.id
        return self._comparer

    def unload(self) -> None:
        """Drop the loaded generator and free GPU memory."""
        with self._model_lock:
            self._release_comparer()

    def generator_availability(self, generator_id: str) -> gens.Availability:
        """Whether a generator can run on this machine."""
        return gens.availability(gens.get_generator(generator_id), api_key_available=vlm_key_available())

    # --- workflow ------------------------------------------------------------

    def is_analyzed(self, frame: Frame, work_size: tuple[int, int] = wf.DEFAULT_WORK_SIZE) -> bool:
        """Whether geometry for ``frame`` is already cached."""
        key = wf.GeometryCache.key(frame, work_size, wf.model_key(self.cfg))
        return self.cache.has(frame, key)

    def analyze(
        self,
        frame: Frame,
        *,
        rig_overrides: dict[str, Any] | None = None,
        work_size: tuple[int, int] = wf.DEFAULT_WORK_SIZE,
    ) -> wf.SceneAnalysis:
        """Scene analysis; depth/segmentation run under the GPU lock on a cache miss."""
        loaded: list[Any] = []

        def _load() -> tuple[Any, Any]:
            loaded.extend(wf.load_conditioning_models(self.cfg))
            return loaded[0], loaded[1]

        if self.is_analyzed(frame, work_size):
            return wf.analyze_frame(
                frame,
                work_size=work_size,
                cache=self.cache,
                model_key=wf.model_key(self.cfg),
                rig_overrides=rig_overrides,
            )
        lock = self._acquire_gpu()
        try:
            return wf.analyze_frame(
                frame,
                work_size=work_size,
                load_models=_load,
                cache=self.cache,
                model_key=wf.model_key(self.cfg),
                rig_overrides=rig_overrides,
            )
        finally:
            loaded.clear()
            free_cuda()
            lock.release()

    def critic(self, use_vlm: bool) -> wf.Critic | None:
        """VLM critic if requested and a key is configured."""
        return wf.make_critic(self.cfg) if use_vlm and vlm_key_available() else None

    def generate(
        self,
        run_id: str,
        frame: Frame,
        analysis: wf.SceneAnalysis,
        *,
        label: str,
        input_kind: str,
        prompt: str,
        slots: dict[str, str] | None,
        seed_offsets: list[int],
        generator: str = gens.DEFAULT_GENERATOR,
        params: dict[str, Any] | None = None,
        negative_prompt: str = "",
        use_vlm: bool = False,
        gates: dict[str, float] | None = None,
        progress: ProgressFn | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[wf.Candidate]:
        """Generate, score, and store one candidate per seed offset."""
        if input_kind not in ("photo", "prior"):
            raise ValueError(f"input_kind must be photo or prior, got {input_kind!r}")
        spec = gens.get_generator(generator)
        resolved = spec.resolve_params(params)
        status = self.generator_availability(spec.id)
        if not status.usable:
            raise RuntimeError(f"{spec.title} cannot run here: {status.reason}")
        negative = negative_prompt if spec.supports_negative else ""
        source_image: Image.Image = frame.load_image() if input_kind == "photo" else analysis.prior
        conditioning = (
            gens.conditioning_from_maps(analysis.disparity, analysis.labels) if spec.needs_conditioning else None
        )
        critic = self.critic(use_vlm)
        out = []
        for offset in seed_offsets:
            if should_stop and should_stop():
                break
            if progress:
                progress(f"{frame.frame_id}: generating {label} with {spec.title} (seed +{offset})")
            lock = None if spec.cloud else self._acquire_gpu()
            try:
                with self._model_lock:
                    # Cloud editors load no weights; keep any local generator loaded.
                    comparer = self._build_comparer(spec) if spec.cloud else self.comparer(spec)
                    image = gens.run_generator(
                        comparer,
                        self.cfg.generation,
                        spec,
                        source_image,
                        prompt,
                        params=resolved,
                        negative_prompt=negative,
                        conditioning=conditioning,
                        work_size=analysis.work_size,
                        seed_offset=int(offset),
                    )
            finally:
                if lock is not None:
                    lock.release()
            if progress:
                progress(f"{frame.frame_id}: scoring {label} (seed +{offset})")
            cand = wf.Candidate(
                label=label,
                input=input_kind,
                prompt=prompt,
                slots=slots,
                seed_offset=int(offset),
                generator=spec.id,
                generator_params=dict(resolved),
                negative_prompt=negative,
            )
            wf.score_candidate(cand, image, frame, analysis, critic=critic, gates=gates)
            self.store.add_candidate(run_id, cand, image)
            out.append(cand)
        return out
