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
        """Whether the Klein pipeline is loaded in this process."""
        return self._comparer is not None

    def comparer(self) -> Any:
        """Lazily build the ``MethodComparer`` (Klein loads on first edit)."""
        if self._comparer is None:
            from aieng.syn_data.image.compare_methods import MethodComparer  # noqa: PLC0415

            self._comparer = MethodComparer.from_config(self.cfg)
        return self._comparer

    def unload(self) -> None:
        """Drop the Klein pipeline and free GPU memory."""
        with self._model_lock:
            if self._comparer is not None:
                self._comparer.unload()
                self._comparer = None
            free_cuda()

    # --- workflow ------------------------------------------------------------

    def is_analyzed(self, frame: Frame, work_size: tuple[int, int] = wf.DEFAULT_WORK_SIZE) -> bool:
        """Whether geometry for ``frame`` is already cached."""
        key = wf.GeometryCache.key(frame, work_size, wf.model_key(self.cfg))
        return self.cache.load(frame, key) is not None

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
        use_vlm: bool = False,
        gates: dict[str, float] | None = None,
        progress: ProgressFn | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[wf.Candidate]:
        """Generate, score, and store one candidate per seed offset."""
        if input_kind not in ("photo", "prior"):
            raise ValueError(f"input_kind must be photo or prior, got {input_kind!r}")
        source_image: Image.Image = frame.load_image() if input_kind == "photo" else analysis.prior
        critic = self.critic(use_vlm)
        out = []
        for offset in seed_offsets:
            if should_stop and should_stop():
                break
            if progress:
                progress(f"{frame.frame_id}: generating {label} (seed +{offset})")
            lock = self._acquire_gpu()
            try:
                with self._model_lock:
                    image = wf.generate_image(
                        self.comparer(),
                        self.cfg.generation,
                        source_image,
                        prompt,
                        work_size=analysis.work_size,
                        seed_offset=int(offset),
                    )
            finally:
                lock.release()
            if progress:
                progress(f"{frame.frame_id}: scoring {label} (seed +{offset})")
            cand = wf.Candidate(label=label, input=input_kind, prompt=prompt, slots=slots, seed_offset=int(offset))
            wf.score_candidate(cand, image, frame, analysis, critic=critic, gates=gates)
            self.store.add_candidate(run_id, cand, image)
            out.append(cand)
        return out
