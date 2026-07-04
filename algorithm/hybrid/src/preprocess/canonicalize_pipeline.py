from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from src.preprocess.event_generation import resolve_event_generation_backend
from src.preprocess.interpolation import resolve_interpolation_backend
from src.preprocess.io_utils import collect_users, ensure_dir, scan_dataset_layout, write_json, write_jsonl
from src.preprocess.progress import print_progress


@dataclass(frozen=True)
class CanonicalizeResolvedRequest:
    raw_root: Path
    canonical_root: Path
    canonical_workspace_root: Path
    indexes_root: Path
    default_frame_size: tuple[int, int]
    default_event_size: tuple[int, int]
    export_masks: bool
    link_mode: str
    overwrite_links: bool
    strict_layout: bool
    skip_nonstandard_sessions: bool
    resolved_annotation_mode: str
    annotation_root: Path | None
    use_raw_ellipse_blink_heuristic: bool
    annotation_backend: str
    groundedsam_root: Path | None
    ultralytics_root: Path | None
    groundedsam2_root: Path | None
    resolved_data_mode: str
    resolved_canonical_name: str
    resolved_frame_source: str
    interpolation_alpha: float
    interpolation_model: str
    interpolation_backend: str | None
    interpolation_target_fps: float | None
    interpolation_fixed_insert: int | None
    interpolation_count_policy: str
    interpolation_max_insert: int | None
    interpolation_timelens_root: Path | None
    interpolation_timelens_checkpoint: Path | None
    interpolation_timelens_device: str
    interpolation_timelens_xl_root: Path | None
    interpolation_timelens_xl_checkpoint: Path | None
    interpolation_timelens_xl_device: str
    event_generation_backend: str
    v2e_root: Path | None
    v2e_device: str
    synthetic_overlap_policy: str
    effective_workers: int
    log_stride: int


class SessionDiscovery:
    def discover(self, raw_root: Path) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for user_dir in collect_users(raw_root):
            try:
                user_id = int(user_dir.name.replace("user", ""))
            except ValueError:
                continue
            for eye in ("left", "right"):
                eye_dir = user_dir / eye
                if not eye_dir.exists():
                    continue
                for raw_session_dir in sorted(eye_dir.glob("session_*_*_*")):
                    sessions.append(
                        {
                            "raw_session_dir": raw_session_dir,
                            "user_id": user_id,
                            "eye": eye,
                        }
                    )
        return sessions


class CanonicalizeJobPlanner:
    def __init__(
        self,
        *,
        canonical_root: Path,
        default_frame_size: tuple[int, int],
        default_event_size: tuple[int, int],
        export_masks: bool,
        link_mode: str,
        overwrite_links: bool,
        strict_layout: bool,
        skip_nonstandard_sessions: bool,
        annotation_mode: str,
        annotation_root: Path | None,
        use_raw_ellipse_blink_heuristic: bool,
        annotation_backend: str,
        groundedsam_root: Path | None,
        ultralytics_root: Path | None,
        groundedsam2_root: Path | None,
        data_mode: str,
        canonical_name: str,
        frame_source: str,
        interpolation_alpha: float,
        interpolation_model: str,
        interpolation_backend: str | None,
        interpolation_target_fps: float | None,
        interpolation_fixed_insert: int | None,
        interpolation_count_policy: str,
        interpolation_max_insert: int | None,
        interpolation_timelens_root: Path | None,
        interpolation_timelens_checkpoint: Path | None,
        interpolation_timelens_device: str,
        interpolation_timelens_xl_root: Path | None,
        interpolation_timelens_xl_checkpoint: Path | None,
        interpolation_timelens_xl_device: str,
        event_generation_backend: str,
        v2e_root: Path | None,
        v2e_device: str,
        synthetic_overlap_policy: str,
    ) -> None:
        self.canonical_root = canonical_root
        self.default_frame_size = default_frame_size
        self.default_event_size = default_event_size
        self.export_masks = export_masks
        self.link_mode = link_mode
        self.overwrite_links = overwrite_links
        self.strict_layout = strict_layout
        self.skip_nonstandard_sessions = skip_nonstandard_sessions
        self.annotation_mode = annotation_mode
        self.annotation_root = annotation_root
        self.use_raw_ellipse_blink_heuristic = use_raw_ellipse_blink_heuristic
        self.annotation_backend = annotation_backend
        self.groundedsam_root = groundedsam_root
        self.ultralytics_root = ultralytics_root
        self.groundedsam2_root = groundedsam2_root
        self.data_mode = data_mode
        self.canonical_name = canonical_name
        self.frame_source = frame_source
        self.interpolation_alpha = interpolation_alpha
        self.interpolation_model = interpolation_model
        self.interpolation_backend = interpolation_backend
        self.interpolation_target_fps = interpolation_target_fps
        self.interpolation_fixed_insert = interpolation_fixed_insert
        self.interpolation_count_policy = interpolation_count_policy
        self.interpolation_max_insert = interpolation_max_insert
        self.interpolation_timelens_root = interpolation_timelens_root
        self.interpolation_timelens_checkpoint = interpolation_timelens_checkpoint
        self.interpolation_timelens_device = interpolation_timelens_device
        self.interpolation_timelens_xl_root = interpolation_timelens_xl_root
        self.interpolation_timelens_xl_checkpoint = interpolation_timelens_xl_checkpoint
        self.interpolation_timelens_xl_device = interpolation_timelens_xl_device
        self.event_generation_backend = event_generation_backend
        self.v2e_root = v2e_root
        self.v2e_device = v2e_device
        self.synthetic_overlap_policy = synthetic_overlap_policy

    def build_jobs(self, discovered_sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for item in discovered_sessions:
            jobs.append(
                {
                    "raw_session_dir": item["raw_session_dir"],
                    "canonical_root": self.canonical_root,
                    "user_id": int(item["user_id"]),
                    "eye": str(item["eye"]),
                    "default_frame_size": self.default_frame_size,
                    "default_event_size": self.default_event_size,
                    "export_masks": self.export_masks,
                    "link_mode": self.link_mode,
                    "overwrite_links": self.overwrite_links,
                    "strict_layout": self.strict_layout,
                    "skip_nonstandard_sessions": self.skip_nonstandard_sessions,
                    "annotation_mode": str(self.annotation_mode),
                    "annotation_root": self.annotation_root,
                    "use_raw_ellipse_blink_heuristic": bool(self.use_raw_ellipse_blink_heuristic),
                    "annotation_backend": str(self.annotation_backend),
                    "groundedsam_root": self.groundedsam_root,
                    "ultralytics_root": self.ultralytics_root,
                    "groundedsam2_root": self.groundedsam2_root,
                    "data_mode": str(self.data_mode),
                    "canonical_name": str(self.canonical_name),
                    "frame_source": str(self.frame_source),
                    "interpolation_alpha": float(self.interpolation_alpha),
                    "interpolation_model": str(self.interpolation_model),
                    "interpolation_backend": None if self.interpolation_backend is None else str(self.interpolation_backend),
                    "interpolation_target_fps": None if self.interpolation_target_fps is None else float(self.interpolation_target_fps),
                    "interpolation_fixed_insert": None if self.interpolation_fixed_insert is None else int(self.interpolation_fixed_insert),
                    "interpolation_count_policy": str(self.interpolation_count_policy),
                    "interpolation_max_insert": None if self.interpolation_max_insert is None else int(self.interpolation_max_insert),
                    "interpolation_timelens_root": self.interpolation_timelens_root,
                    "interpolation_timelens_checkpoint": self.interpolation_timelens_checkpoint,
                    "interpolation_timelens_device": str(self.interpolation_timelens_device),
                    "interpolation_timelens_xl_root": self.interpolation_timelens_xl_root,
                    "interpolation_timelens_xl_checkpoint": self.interpolation_timelens_xl_checkpoint,
                    "interpolation_timelens_xl_device": str(self.interpolation_timelens_xl_device),
                    "event_generation_backend": str(self.event_generation_backend),
                    "v2e_root": self.v2e_root,
                    "v2e_device": str(self.v2e_device),
                    "synthetic_overlap_policy": str(self.synthetic_overlap_policy),
                }
            )
        return jobs


class CanonicalizeResultCollector:
    def __init__(self, *, started_at: float, total_jobs: int, log_stride: int) -> None:
        self.started_at = started_at
        self.total_jobs = int(total_jobs)
        self.log_stride = max(int(log_stride), 1)
        self.session_rows: list[dict[str, Any]] = []
        self.skipped_rows: list[dict[str, Any]] = []
        self.processed = 0
        self.n_ok = 0
        self.n_skipped = 0

    def consume(self, result: dict[str, Any] | None) -> None:
        self.processed += 1
        if result is not None:
            self.session_rows.append(result)
            if result.get("skipped"):
                self.skipped_rows.append(result)
                self.n_skipped += 1
            else:
                self.n_ok += 1
        if self.processed == 1 or self.processed % self.log_stride == 0 or self.processed == self.total_jobs:
            print_progress(
                name="canonicalize",
                status="progress",
                started_at=self.started_at,
                completed=self.processed,
                total=self.total_jobs,
                message="session processed",
                n_ok=self.n_ok,
                n_skipped=self.n_skipped,
            )


class CanonicalizeSummaryWriter:
    def __init__(
        self,
        *,
        raw_root: Path,
        canonical_root: Path,
        canonical_workspace_root: Path,
        indexes_root: Path,
    ) -> None:
        self.raw_root = raw_root
        self.canonical_root = canonical_root
        self.canonical_workspace_root = canonical_workspace_root
        self.indexes_root = indexes_root

    def write_preflight(self, preflight_rows: list[dict[str, Any]]) -> None:
        write_jsonl(preflight_rows, self.indexes_root / "preflight_scan.jsonl")

    def write_session_indexes(
        self,
        *,
        session_rows: list[dict[str, Any]],
        skipped_rows: list[dict[str, Any]],
    ) -> None:
        write_jsonl(session_rows, self.indexes_root / "sessions.jsonl")
        write_jsonl(skipped_rows, self.indexes_root / "canonical_skipped_sessions.jsonl")

    def build_summary(
        self,
        *,
        session_rows: list[dict[str, Any]],
        skipped_rows: list[dict[str, Any]],
        resolved_annotation_mode: str,
        annotation_root: Path | None,
        use_raw_ellipse_blink_heuristic: bool,
        annotation_backend: str,
        groundedsam_root: Path | None,
        ultralytics_root: Path | None,
        groundedsam2_root: Path | None,
        resolved_data_mode: str,
        resolved_canonical_name: str,
        resolved_frame_source: str,
        interpolation_alpha: float,
        interpolation_model: str,
        interpolation_backend: str | None,
        interpolation_target_fps: float | None,
        interpolation_fixed_insert: int | None,
        interpolation_count_policy: str,
        interpolation_max_insert: int | None,
        interpolation_timelens_root: Path | None,
        interpolation_timelens_checkpoint: Path | None,
        interpolation_timelens_device: str,
        interpolation_timelens_xl_root: Path | None,
        interpolation_timelens_xl_checkpoint: Path | None,
        interpolation_timelens_xl_device: str,
        event_generation_backend: str,
        v2e_root: Path | None,
        v2e_device: str,
        synthetic_overlap_policy: str,
        effective_workers: int,
    ) -> dict[str, Any]:
        return {
            "raw_root": str(self.raw_root),
            "canonical_root": str(self.canonical_root),
            "canonical_workspace_root": str(self.canonical_workspace_root),
            "indexes_root": str(self.indexes_root),
            "n_sessions_total": len(session_rows),
            "n_sessions_ok": sum(1 for row in session_rows if not row.get("skipped")),
            "n_sessions_skipped": len(skipped_rows),
            "annotation_mode": str(resolved_annotation_mode),
            "annotation_root": None if annotation_root is None else str(annotation_root),
            "use_raw_ellipse_blink_heuristic": bool(use_raw_ellipse_blink_heuristic),
            "annotation_backend": str(annotation_backend),
            "groundedsam_root": None if groundedsam_root is None else str(groundedsam_root),
            "ultralytics_root": None if ultralytics_root is None else str(ultralytics_root),
            "groundedsam2_root": None if groundedsam2_root is None else str(groundedsam2_root),
            "data_mode": str(resolved_data_mode),
            "canonical_name": str(resolved_canonical_name),
            "frame_source": str(resolved_frame_source),
            "interpolation_alpha": float(interpolation_alpha),
            "interpolation_model": str(interpolation_model),
            "interpolation_backend": resolve_interpolation_backend(
                interpolation_backend=interpolation_backend,
                interpolation_model=interpolation_model,
            ),
            "interpolation_target_fps": None if interpolation_target_fps is None else float(interpolation_target_fps),
            "interpolation_fixed_insert": None if interpolation_fixed_insert is None else int(interpolation_fixed_insert),
            "interpolation_count_policy": str(interpolation_count_policy),
            "interpolation_max_insert": None if interpolation_max_insert is None else int(interpolation_max_insert),
            "interpolation_timelens_root": None if interpolation_timelens_root is None else str(Path(interpolation_timelens_root).resolve()),
            "interpolation_timelens_checkpoint": None if interpolation_timelens_checkpoint is None else str(Path(interpolation_timelens_checkpoint).resolve()),
            "interpolation_timelens_device": str(interpolation_timelens_device),
            "interpolation_timelens_xl_root": None if interpolation_timelens_xl_root is None else str(Path(interpolation_timelens_xl_root).resolve()),
            "interpolation_timelens_xl_checkpoint": None if interpolation_timelens_xl_checkpoint is None else str(Path(interpolation_timelens_xl_checkpoint).resolve()),
            "interpolation_timelens_xl_device": str(interpolation_timelens_xl_device),
            "event_generation_backend": resolve_event_generation_backend(event_generation_backend),
            "v2e_root": None if v2e_root is None else str(Path(v2e_root).resolve()),
            "v2e_device": str(v2e_device),
            "synthetic_overlap_policy": str(synthetic_overlap_policy),
            "num_workers": int(effective_workers),
        }

    def write_summary(self, summary: dict[str, Any]) -> None:
        write_json(summary, self.indexes_root / "canonical_summary.json")


class CanonicalizeProgressReporter:
    def __init__(self, *, started_at: float) -> None:
        self.started_at = float(started_at)

    def start(self, request: CanonicalizeResolvedRequest) -> None:
        print_progress(
            name="canonicalize",
            status="start",
            started_at=self.started_at,
            total=0,
            message="canonicalization started",
            raw_root=str(request.raw_root),
            canonical_root=str(request.canonical_root),
            canonical_workspace_root=str(request.canonical_workspace_root),
            indexes_root=str(request.indexes_root),
            annotation_mode=str(request.resolved_annotation_mode),
            annotation_root=None if request.annotation_root is None else str(request.annotation_root),
            use_raw_ellipse_blink_heuristic=bool(request.use_raw_ellipse_blink_heuristic),
            annotation_backend=str(request.annotation_backend),
            groundedsam_root=None if request.groundedsam_root is None else str(request.groundedsam_root),
            ultralytics_root=None if request.ultralytics_root is None else str(request.ultralytics_root),
            groundedsam2_root=None if request.groundedsam2_root is None else str(request.groundedsam2_root),
            data_mode=str(request.resolved_data_mode),
            canonical_name=str(request.resolved_canonical_name),
            frame_source=str(request.resolved_frame_source),
            interpolation_alpha=float(request.interpolation_alpha),
            interpolation_model=str(request.interpolation_model),
            interpolation_backend=None if request.interpolation_backend is None else str(request.interpolation_backend),
            interpolation_target_fps=None if request.interpolation_target_fps is None else float(request.interpolation_target_fps),
            interpolation_fixed_insert=None if request.interpolation_fixed_insert is None else int(request.interpolation_fixed_insert),
            interpolation_count_policy=str(request.interpolation_count_policy),
            interpolation_max_insert=None if request.interpolation_max_insert is None else int(request.interpolation_max_insert),
            interpolation_timelens_root=None if request.interpolation_timelens_root is None else str(Path(request.interpolation_timelens_root).resolve()),
            interpolation_timelens_checkpoint=None if request.interpolation_timelens_checkpoint is None else str(Path(request.interpolation_timelens_checkpoint).resolve()),
            interpolation_timelens_device=str(request.interpolation_timelens_device),
            interpolation_timelens_xl_root=None if request.interpolation_timelens_xl_root is None else str(Path(request.interpolation_timelens_xl_root).resolve()),
            interpolation_timelens_xl_checkpoint=None if request.interpolation_timelens_xl_checkpoint is None else str(Path(request.interpolation_timelens_xl_checkpoint).resolve()),
            interpolation_timelens_xl_device=str(request.interpolation_timelens_xl_device),
            event_generation_backend=str(request.event_generation_backend),
            v2e_root=None if request.v2e_root is None else str(Path(request.v2e_root).resolve()),
            v2e_device=str(request.v2e_device),
            synthetic_overlap_policy=str(request.synthetic_overlap_policy),
            num_workers=int(request.effective_workers),
        )

    def preflight_done(self, *, n_preflight_rows: int) -> None:
        print_progress(
            name="canonicalize",
            status="preflight_done",
            started_at=self.started_at,
            total=0,
            message="preflight scan written",
            n_preflight_rows=int(n_preflight_rows),
        )

    def sessions_discovered(self, *, total_jobs: int) -> None:
        print_progress(
            name="canonicalize",
            status="sessions_discovered",
            started_at=self.started_at,
            completed=0,
            total=int(total_jobs),
            message="session job list prepared",
            n_session_jobs=int(total_jobs),
        )

    def done(self, *, total_jobs: int, summary: dict[str, Any]) -> None:
        print_progress(
            name="canonicalize",
            status="done",
            started_at=self.started_at,
            completed=int(total_jobs),
            total=int(total_jobs),
            message="canonicalization complete",
            n_sessions_total=summary["n_sessions_total"],
            n_sessions_ok=summary["n_sessions_ok"],
            n_sessions_skipped=summary["n_sessions_skipped"],
        )


class CanonicalSessionProcessor:
    def __init__(self, *, session_callable: Any) -> None:
        self.session_callable = session_callable

    def process_job(self, job: dict[str, Any]) -> dict[str, Any] | None:
        return self.session_callable(**job)

    def process_jobs(
        self,
        *,
        session_jobs: list[dict[str, Any]],
        effective_workers: int,
        collector: CanonicalizeResultCollector,
        started_at: float,
        total_jobs: int,
        print_progress_fn: Any,
    ) -> None:
        if effective_workers > 1 and total_jobs > 0:
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                futures = {executor.submit(self.process_job, job): job for job in session_jobs}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as exc:
                        print_progress_fn(
                            name="canonicalize",
                            status="error",
                            started_at=started_at,
                            completed=collector.processed,
                            total=total_jobs,
                            message="canonicalize worker failed",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        raise
                    collector.consume(result)
            return

        for job in session_jobs:
            collector.consume(self.process_job(job))


class CanonicalizeDatasetRunner:
    def __init__(self, *, request: CanonicalizeResolvedRequest, session_callable: Any) -> None:
        self.request = request
        self.session_callable = session_callable

    @staticmethod
    def _session_sort_key(row: dict[str, Any]) -> tuple[int, str, str, str]:
        return (
            int(row.get("user_id", -1)),
            str(row.get("eye", "")),
            str(row.get("session_code", "")),
            str(row.get("session_key", "")),
        )

    def run(self) -> dict[str, Any]:
        ensure_dir(self.request.indexes_root)
        ensure_dir(self.request.canonical_root / "sessions")

        started_at = time.perf_counter()
        reporter = CanonicalizeProgressReporter(started_at=started_at)
        summary_writer = CanonicalizeSummaryWriter(
            raw_root=self.request.raw_root,
            canonical_root=self.request.canonical_root,
            canonical_workspace_root=self.request.canonical_workspace_root,
            indexes_root=self.request.indexes_root,
        )
        reporter.start(self.request)

        preflight_rows = scan_dataset_layout(self.request.raw_root)
        summary_writer.write_preflight(preflight_rows)
        reporter.preflight_done(n_preflight_rows=len(preflight_rows))

        discovered_sessions = SessionDiscovery().discover(self.request.raw_root)
        session_jobs = CanonicalizeJobPlanner(
            canonical_root=self.request.canonical_root,
            default_frame_size=self.request.default_frame_size,
            default_event_size=self.request.default_event_size,
            export_masks=self.request.export_masks,
            link_mode=self.request.link_mode,
            overwrite_links=self.request.overwrite_links,
            strict_layout=self.request.strict_layout,
            skip_nonstandard_sessions=self.request.skip_nonstandard_sessions,
            annotation_mode=str(self.request.resolved_annotation_mode),
            annotation_root=self.request.annotation_root,
            use_raw_ellipse_blink_heuristic=bool(self.request.use_raw_ellipse_blink_heuristic),
            annotation_backend=str(self.request.annotation_backend),
            groundedsam_root=self.request.groundedsam_root,
            ultralytics_root=self.request.ultralytics_root,
            groundedsam2_root=self.request.groundedsam2_root,
            data_mode=str(self.request.resolved_data_mode),
            canonical_name=str(self.request.resolved_canonical_name),
            frame_source=str(self.request.resolved_frame_source),
            interpolation_alpha=float(self.request.interpolation_alpha),
            interpolation_model=str(self.request.interpolation_model),
            interpolation_backend=None if self.request.interpolation_backend is None else str(self.request.interpolation_backend),
            interpolation_target_fps=None if self.request.interpolation_target_fps is None else float(self.request.interpolation_target_fps),
            interpolation_fixed_insert=None if self.request.interpolation_fixed_insert is None else int(self.request.interpolation_fixed_insert),
            interpolation_count_policy=str(self.request.interpolation_count_policy),
            interpolation_max_insert=None if self.request.interpolation_max_insert is None else int(self.request.interpolation_max_insert),
            interpolation_timelens_root=self.request.interpolation_timelens_root,
            interpolation_timelens_checkpoint=self.request.interpolation_timelens_checkpoint,
            interpolation_timelens_device=str(self.request.interpolation_timelens_device),
            interpolation_timelens_xl_root=self.request.interpolation_timelens_xl_root,
            interpolation_timelens_xl_checkpoint=self.request.interpolation_timelens_xl_checkpoint,
            interpolation_timelens_xl_device=str(self.request.interpolation_timelens_xl_device),
            event_generation_backend=str(self.request.event_generation_backend),
            v2e_root=self.request.v2e_root,
            v2e_device=str(self.request.v2e_device),
            synthetic_overlap_policy=str(self.request.synthetic_overlap_policy),
        ).build_jobs(discovered_sessions)

        total_jobs = len(session_jobs)
        reporter.sessions_discovered(total_jobs=total_jobs)

        collector = CanonicalizeResultCollector(
            started_at=started_at,
            total_jobs=total_jobs,
            log_stride=self.request.log_stride,
        )
        CanonicalSessionProcessor(session_callable=self.session_callable).process_jobs(
            session_jobs=session_jobs,
            effective_workers=self.request.effective_workers,
            collector=collector,
            started_at=started_at,
            total_jobs=total_jobs,
            print_progress_fn=print_progress,
        )

        session_rows = sorted(collector.session_rows, key=self._session_sort_key)
        skipped_rows = sorted(collector.skipped_rows, key=self._session_sort_key)
        summary_writer.write_session_indexes(session_rows=session_rows, skipped_rows=skipped_rows)
        summary = summary_writer.build_summary(
            session_rows=session_rows,
            skipped_rows=skipped_rows,
            resolved_annotation_mode=str(self.request.resolved_annotation_mode),
            annotation_root=self.request.annotation_root,
            use_raw_ellipse_blink_heuristic=bool(self.request.use_raw_ellipse_blink_heuristic),
            annotation_backend=str(self.request.annotation_backend),
            groundedsam_root=self.request.groundedsam_root,
            ultralytics_root=self.request.ultralytics_root,
            groundedsam2_root=self.request.groundedsam2_root,
            resolved_data_mode=str(self.request.resolved_data_mode),
            resolved_canonical_name=str(self.request.resolved_canonical_name),
            resolved_frame_source=str(self.request.resolved_frame_source),
            interpolation_alpha=float(self.request.interpolation_alpha),
            interpolation_model=str(self.request.interpolation_model),
            interpolation_backend=None if self.request.interpolation_backend is None else str(self.request.interpolation_backend),
            interpolation_target_fps=None if self.request.interpolation_target_fps is None else float(self.request.interpolation_target_fps),
            interpolation_fixed_insert=None if self.request.interpolation_fixed_insert is None else int(self.request.interpolation_fixed_insert),
            interpolation_count_policy=str(self.request.interpolation_count_policy),
            interpolation_max_insert=None if self.request.interpolation_max_insert is None else int(self.request.interpolation_max_insert),
            interpolation_timelens_root=self.request.interpolation_timelens_root,
            interpolation_timelens_checkpoint=self.request.interpolation_timelens_checkpoint,
            interpolation_timelens_device=str(self.request.interpolation_timelens_device),
            interpolation_timelens_xl_root=self.request.interpolation_timelens_xl_root,
            interpolation_timelens_xl_checkpoint=self.request.interpolation_timelens_xl_checkpoint,
            interpolation_timelens_xl_device=str(self.request.interpolation_timelens_xl_device),
            event_generation_backend=str(self.request.event_generation_backend),
            v2e_root=self.request.v2e_root,
            v2e_device=str(self.request.v2e_device),
            synthetic_overlap_policy=str(self.request.synthetic_overlap_policy),
            effective_workers=self.request.effective_workers,
        )
        summary_writer.write_summary(summary)
        reporter.done(total_jobs=total_jobs, summary=summary)
        return summary


class CanonicalSessionArtifactWriter:
    def __init__(self, *, canonical_root: Path, relativize_fn: Any, session_package_builder: Any) -> None:
        self.canonical_root = canonical_root
        self.relativize_fn = relativize_fn
        self.session_package_builder = session_package_builder

    def write(
        self,
        *,
        session_key: str,
        user_id: int,
        eye: str,
        session_code: str,
        raw_session_dir: Path,
        session_dir: Path,
        frame_records: list[Any],
        label_rows: list[dict[str, Any]],
        eye_region: Any,
        dst_event_npz: Path,
        event_origin: str,
        layout: Any,
        annotation_mode: str,
        annotation_source_kind: str,
        annotation_parse_report: dict[str, Any],
        blink_candidate_report: dict[str, Any],
        data_mode: str,
        canonical_name: str,
        frame_source: str,
        protocol_session_index: int | None,
        session_motion_regime_prior: dict[str, Any],
        use_raw_ellipse_blink_heuristic: bool,
        interpolation_summary: dict[str, Any] | None,
        sensor_width: int,
        sensor_height: int,
    ) -> dict[str, Any]:
        frame_index_path = session_dir / "labels" / "frame_index.jsonl"
        ann_path = session_dir / "labels" / "frame_annotations.jsonl"
        session_package_path = session_dir / "labels" / "session_package.json"
        interpolation_ref = (
            None
            if interpolation_summary is None
            else self.relativize_fn(self.canonical_root, session_dir / "labels" / "interpolation_index.json")
        )

        write_jsonl([rec.to_dict() for rec in frame_records], frame_index_path)
        write_jsonl(label_rows, ann_path)

        source_size = [int(eye_region.w), int(eye_region.h)]
        eye_region_payload = {
            "session_key": session_key,
            "annotation_store_path": self.relativize_fn(self.canonical_root, ann_path),
            "eye_region_xywh": eye_region.to_list(),
            "sensor_size_wh": [sensor_width, sensor_height],
            "frame_source_size_wh": source_size,
            "event_source_size_wh": source_size,
            "n_annotations": len(label_rows),
            "annotation_parse_report": annotation_parse_report,
            "blink_candidate_report": blink_candidate_report,
            "annotation_mode": annotation_mode,
            "annotation_source_kind": annotation_source_kind,
            "data_mode": str(data_mode),
            "canonical_name": str(canonical_name),
            "frame_source": str(frame_source),
            "interpolation_ref": interpolation_ref,
            "interpolation_summary": interpolation_summary,
            "layout": layout.to_dict(),
            "protocol_session_index": protocol_session_index,
            "session_motion_regime_prior": session_motion_regime_prior,
            "session_package_path": self.relativize_fn(self.canonical_root, session_package_path),
        }
        write_json(eye_region_payload, session_dir / "labels" / "eye_region.json")

        session_package = self.session_package_builder(
            session_key=session_key,
            user_id=user_id,
            eye=eye,
            session_code=session_code,
            data_mode=data_mode,
            canonical_name=canonical_name,
            frame_source=frame_source,
            protocol_session_index=protocol_session_index,
            sensor_size_wh=[sensor_width, sensor_height],
            eye_region_xywh=eye_region.to_list(),
            frame_source_size_wh=source_size,
            event_source_size_wh=source_size,
            events_npz=self.relativize_fn(self.canonical_root, dst_event_npz),
            frame_index_path=self.relativize_fn(self.canonical_root, frame_index_path),
            annotation_store_path=self.relativize_fn(self.canonical_root, ann_path),
            session_motion_regime_prior=session_motion_regime_prior,
            n_frames=len(frame_records),
            n_labelled_frames=len(label_rows),
            interpolation_ref=interpolation_ref,
            interpolation_summary=interpolation_summary,
        )
        write_json(session_package, session_package_path)

        meta = {
            "user_id": user_id,
            "eye": eye,
            "session_code": session_code,
            "session_dir_name": raw_session_dir.name,
            "session_key": session_key,
            "protocol_session_index": protocol_session_index,
            "session_motion_regime_prior": session_motion_regime_prior,
            "session_package_path": self.relativize_fn(self.canonical_root, session_package_path),
            "sensor_size_wh": [sensor_width, sensor_height],
            "frame_source_size_wh": source_size,
            "event_source_size_wh": source_size,
            "eye_region_xywh": eye_region.to_list(),
            "n_frames": len(frame_records),
            "n_labelled_frames": len(label_rows),
            "annotation_store_path": self.relativize_fn(self.canonical_root, ann_path),
            "events_npz": self.relativize_fn(self.canonical_root, dst_event_npz),
            "event_origin": event_origin,
            "raw_session_dir": str(raw_session_dir.resolve()),
            "frames_dir": self.relativize_fn(self.canonical_root, session_dir / "frames"),
            "layout_warnings": list(layout.warnings),
            "annotation_csv": str(layout.annotation_csv.resolve()) if layout.annotation_csv else None,
            "annotation_parse_report": annotation_parse_report,
            "blink_candidate_report": blink_candidate_report,
            "annotation_mode": annotation_mode,
            "annotation_source_kind": annotation_source_kind,
            "session_official": bool(layout.is_official_session),
            "use_raw_ellipse_blink_heuristic": bool(use_raw_ellipse_blink_heuristic),
            "data_mode": str(data_mode),
            "canonical_name": str(canonical_name),
            "frame_source": str(frame_source),
            "interpolation_ref": interpolation_ref,
            "interpolation_summary": interpolation_summary,
        }
        write_json(meta, session_dir / "meta.json")
        return meta


__all__ = [
    "CanonicalSessionArtifactWriter",
    "CanonicalizeDatasetRunner",
    "CanonicalizeJobPlanner",
    "CanonicalizeProgressReporter",
    "CanonicalizeResultCollector",
    "CanonicalizeResolvedRequest",
    "CanonicalSessionProcessor",
    "CanonicalizeSummaryWriter",
    "SessionDiscovery",
]
