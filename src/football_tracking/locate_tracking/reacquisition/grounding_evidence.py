"""Grounding compatibility evidence from M5 grounding artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from football_tracking.locate_tracking.artifacts.mot_schemas import MotTrackObservation
from football_tracking.locate_tracking.artifacts.track_index import FrameTrackIndex
from football_tracking.locate_tracking.association.geometry import (
    bbox_iou,
    center_similarity,
    coverage_of_left,
)
from football_tracking.locate_tracking.grounding.schemas import GroundingResult
from football_tracking.locate_tracking.reacquisition.schemas import (
    EvidenceScore,
    GateResult,
    ReacquisitionCandidate,
    ReacquisitionConfig,
)


def _load_grounding_result(path: str | Path) -> GroundingResult:
    resolved = Path(path)
    return GroundingResult.from_dict(json.loads(resolved.read_text(encoding="utf-8")))


def grounding_result_paths_from_manifest(manifest_path: str | Path) -> tuple[Path, ...]:
    resolved = Path(manifest_path)
    if not resolved.is_file():
        return ()
    data = json.loads(resolved.read_text(encoding="utf-8"))
    paths: list[Path] = []
    for request in data.get("executed_requests", ()):
        for frame in request.get("frames", ()):
            result_path = frame.get("grounding_result_path")
            if result_path:
                paths.append(Path(result_path))
    return tuple(paths)


def grounding_evidence(
    *,
    candidate: ReacquisitionCandidate,
    all_observations: tuple[MotTrackObservation, ...],
    grounding_result_paths: tuple[str | Path, ...],
) -> EvidenceScore:
    if not grounding_result_paths:
        return EvidenceScore(
            name="grounding",
            score=None,
            data_available=False,
            reason="no_grounding_results_available",
        )
    index = FrameTrackIndex.from_observations(all_observations)
    matches: list[dict[str, float | int | str]] = []
    errors: list[str] = []
    for path in grounding_result_paths:
        try:
            result = _load_grounding_result(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: {exc}")
            continue
        frame_index_raw = result.request.image_path.stem.replace("frame_", "")
        try:
            frame_index = int(frame_index_raw)
        except ValueError:
            continue
        candidate_row = next(
            (row for row in index.get_frame(frame_index) if row.track_id == candidate.raw_track_id),
            None,
        )
        if candidate_row is None:
            continue
        for box in result.boxes:
            iou = bbox_iou(box.bbox_xyxy, candidate_row.bbox_xyxy)
            coverage = coverage_of_left(candidate_row.bbox_xyxy, box.bbox_xyxy)
            center = center_similarity(
                box.bbox_xyxy,
                candidate_row.bbox_xyxy,
                result.image_width,
                result.image_height,
            )
            score = (0.55 * iou) + (0.30 * coverage) + (0.15 * center)
            matches.append(
                {
                    "frame_index": frame_index,
                    "grounding_result_path": str(path),
                    "iou": iou,
                    "track_coverage": coverage,
                    "center_similarity": center,
                    "score": score,
                }
            )
    if not matches:
        return EvidenceScore(
            name="grounding",
            score=0.0,
            data_available=True,
            reason="no_grounding_box_matched_candidate",
            details={"errors": errors},
        )
    best_by_frame: dict[int, float] = {}
    for match in matches:
        frame_index = int(match["frame_index"])
        best_by_frame[frame_index] = max(best_by_frame.get(frame_index, 0.0), float(match["score"]))
    score = sum(best_by_frame.values()) / len(best_by_frame)
    return EvidenceScore(
        name="grounding",
        score=max(0.0, min(1.0, score)),
        data_available=True,
        reason="grounding_track_compatibility",
        details={"matched_frames": sorted(best_by_frame), "matches": matches[:20]},
    )


def grounding_gate(evidence: EvidenceScore, config: ReacquisitionConfig) -> GateResult:
    if not evidence.data_available or evidence.score is None:
        return GateResult(
            gate_name="grounding",
            passed=not config.require_grounding_support,
            score=None,
            threshold=config.min_grounding_score,
            reason="grounding_unavailable",
        )
    passed = evidence.score >= config.min_grounding_score
    return GateResult(
        gate_name="grounding",
        passed=passed,
        score=evidence.score,
        threshold=config.min_grounding_score,
        reason="grounding_supported" if passed else "below_minimum_grounding_score",
        metadata=dict(evidence.details),
    )
