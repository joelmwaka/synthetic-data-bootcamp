"""Batch worker for Synthetic Data Studio.

Polls the job queue in ``outputs/<dataset>/studio/jobs.sqlite`` and runs one
job at a time. Start it from the app's Batch page, or manually::

    cd synthetic_data_studio
    uv run python worker.py --dataset l4r_nlb --hardware gpu_l4

``--once`` processes queued jobs and exits when the queue is empty.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from pathlib import Path
from types import FrameType


sys.path.insert(0, str(Path(__file__).resolve().parent))

from studio.batch import run_job  # noqa: E402
from studio.engine import Engine, load_cfg  # noqa: E402
from studio.jobs import Job, JobQueue  # noqa: E402


HEARTBEAT_EVERY_S = 5.0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="l4r_nlb")
    parser.add_argument("--hardware", default="gpu_l4")
    parser.add_argument("--once", action="store_true", help="exit when the queue is empty")
    parser.add_argument("--poll", type=float, default=3.0, help="seconds between queue polls")
    return parser.parse_args(argv)


def process_job(job: Job, queue: JobQueue, engine: Engine, stop: threading.Event) -> str:
    """Run one claimed job and record its final status."""
    try:
        result = run_job(
            job.id,
            job.spec,
            engine,
            report=lambda p: queue.update_progress(job.id, p),
            should_cancel=lambda: stop.is_set() or queue.cancel_requested(job.id),
        )
    except Exception as exc:  # noqa: BLE001 - keep the worker alive
        queue.finish(job.id, status="failed", error=f"{type(exc).__name__}: {exc}")
    else:
        failed_all = result["frame_errors"] and len(result["frame_errors"]) == result["frames_total"]
        if result["cancelled"]:
            queue.finish(job.id, status="cancelled")
        elif failed_all:
            queue.finish(job.id, status="failed", error="every frame failed; see progress.frame_errors")
        else:
            queue.finish(job.id, status="done")
    finally:
        engine.unload()
    return queue.get(job.id).status


def main(argv: list[str] | None = None) -> int:
    """Run the worker loop."""
    args = _parse_args(argv)
    engine = Engine(load_cfg(args.dataset, args.hardware), gpu_timeout=600.0)
    queue = JobQueue(engine.root / "jobs.sqlite")
    pid = os.getpid()
    stop = threading.Event()
    current: dict[str, int | None] = {"job": None}

    def _signal(signum: int, _frame: FrameType | None) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, _signal)
    signal.signal(signal.SIGINT, _signal)

    def _beat() -> None:
        while not stop.wait(HEARTBEAT_EVERY_S):
            queue.heartbeat(pid, current["job"])

    queue.heartbeat(pid)
    threading.Thread(target=_beat, daemon=True).start()
    recovered = queue.recover_orphans()
    print(f"[worker {pid}] dataset={args.dataset} hardware={args.hardware} recovered_orphans={recovered}", flush=True)
    try:
        while not stop.is_set():
            job = queue.claim_next(pid)
            if job is None:
                if args.once:
                    break
                stop.wait(args.poll)
                continue
            if job.spec.get("dataset") != args.dataset or job.spec.get("hardware") != args.hardware:
                queue.finish(job.id, status="failed", error="job dataset/hardware does not match this worker")
                continue
            current["job"] = job.id
            queue.heartbeat(pid, job.id)
            print(f"[worker {pid}] job {job.id}: {len(job.spec['frame_ids'])} frames", flush=True)
            status = process_job(job, queue, engine, stop)
            current["job"] = None
            print(f"[worker {pid}] job {job.id} finished: {status}", flush=True)
    finally:
        stop.set()
        queue.remove_worker(pid)
        engine.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
