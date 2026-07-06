"""Offline appearance verification service for M3 semantic candidates."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from football_tracking.locate_tracking.appearance.backend import AppearanceEmbeddingProvider
from football_tracking.locate_tracking.appearance.cache import AppearanceEmbeddingCache
from football_tracking.locate_tracking.appearance.config import AppearanceVerificationConfig
from football_tracking.locate_tracking.appearance.crop_extractor import (
    CropExtractionError,
    TrackCrop,
    TrackCropExtractor,
)
from football_tracking.locate_tracking.appearance.crop_selection import select_representative_crops
from football_tracking.locate_tracking.appearance.mock_backend import (
    MockAppearanceEmbeddingProvider,
)
from football_tracking.locate_tracking.appearance.prototype_bank import (
    PrototypeBuildError,
    build_track_prototype,
)
from football_tracking.locate_tracking.appearance.schemas import (
    AppearanceRuntimeInfo,
    AppearanceVerificationResult,
    TrackEmbeddingSample,
)
from football_tracking.locate_tracking.appearance.ultralytics_backend import (
    UltralyticsAppearanceEmbeddingProvider,
)
from football_tracking.locate_tracking.appearance.verifier import score_appearance_prototypes
from football_tracking.locate_tracking.artifacts.mot_reader import read_mot_track_file
from football_tracking.locate_tracking.artifacts.track_index import FrameTrackIndex
from football_tracking.locate_tracking.fusion.decision_policy import decide_fused_resolution
from football_tracking.locate_tracking.fusion.schemas import FusionResult
from football_tracking.locate_tracking.fusion.score_fusion import fuse_scores
from football_tracking.locate_tracking.semantic_memory.schemas import (
    CandidateSemanticMemory,
    SemanticMemory,
)
from football_tracking.locate_tracking.semantic_memory.serialization import load_semantic_memory
from football_tracking.locate_tracking.video.frame_extractor import extract_video_frame


class AppearanceVerificationServiceError(RuntimeError):
    """Raised when appearance verification cannot complete."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: dict[str, Any], *, overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        raise AppearanceVerificationServiceError(f"Output exists and overwrite=false: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def build_appearance_provider(config: AppearanceVerificationConfig) -> AppearanceEmbeddingProvider:
    if config.backend_name == "mock":
        return MockAppearanceEmbeddingProvider(model_id=config.model_id, normalize=config.normalize)
    return UltralyticsAppearanceEmbeddingProvider(
        model_id=config.model_id,
        device=config.device,
        batch_size=config.batch_size,
        normalize=config.normalize,
    )


def _candidate_frames(candidate: CandidateSemanticMemory) -> tuple[int, ...]:
    frames = candidate.frames_with_grounding_match or candidate.frames_present
    return tuple(sorted(set(int(frame) for frame in frames)))


class AppearanceVerificationService:
    def __init__(
        self,
        *,
        config: AppearanceVerificationConfig,
        provider: AppearanceEmbeddingProvider | None = None,
    ) -> None:
        self.config = config
        self.provider = provider or build_appearance_provider(config)
        self.cache = AppearanceEmbeddingCache(
            config.cache_directory,
            enabled=config.cache_enabled,
            overwrite=config.overwrite,
        )
        self.crop_extractor = TrackCropExtractor(config.crop_quality)

    def _save_crop(self, crop: TrackCrop, output_root: Path) -> TrackCrop:
        if not self.config.save_crops:
            return crop
        import cv2  # type: ignore[import-not-found]

        path = (
            output_root
            / "crops"
            / f"track_{crop.reference.raw_track_id:06d}"
            / f"frame_{crop.reference.frame_index:06d}.jpg"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), crop.image):
            raise AppearanceVerificationServiceError(f"Could not write crop: {path}")
        updated_ref = type(crop.reference)(
            raw_track_id=crop.reference.raw_track_id,
            frame_index=crop.reference.frame_index,
            source_video=crop.reference.source_video,
            raw_bbox_xyxy=crop.reference.raw_bbox_xyxy,
            clipped_bbox_xyxy=crop.reference.clipped_bbox_xyxy,
            crop_width=crop.reference.crop_width,
            crop_height=crop.reference.crop_height,
            crop_path=str(path),
            quality_metrics=crop.reference.quality_metrics,
        )
        return TrackCrop(reference=updated_ref, image=crop.image)

    def _extract_candidate_crops(
        self,
        *,
        semantic_memory: SemanticMemory,
        source_video: Path,
        tracks_path: Path,
        output_root: Path,
    ) -> dict[int, tuple[TrackCrop, ...]]:
        mot_file = read_mot_track_file(tracks_path)
        track_index = FrameTrackIndex.from_observations(mot_file.observations)
        frames_needed = sorted(
            {
                frame
                for candidate in semantic_memory.candidate_memories
                for frame in _candidate_frames(candidate)
            }
        )
        extracted_frames = {
            frame_index: extract_video_frame(source_video, frame_index).image
            for frame_index in frames_needed
        }
        crops_by_track: dict[int, list[TrackCrop]] = {
            candidate.raw_track_id: [] for candidate in semantic_memory.candidate_memories
        }
        for candidate in semantic_memory.candidate_memories:
            for frame_index in _candidate_frames(candidate):
                observations = {
                    observation.track_id: observation
                    for observation in track_index.get_frame(frame_index)
                }
                observation = observations.get(candidate.raw_track_id)
                if observation is None:
                    continue
                try:
                    crop = self.crop_extractor.extract(
                        frame=extracted_frames[frame_index],
                        observation=observation,
                        source_video=source_video,
                    )
                except CropExtractionError:
                    continue
                crops_by_track[candidate.raw_track_id].append(self._save_crop(crop, output_root))
        return {key: tuple(value) for key, value in crops_by_track.items()}

    def _embed_selected_crops(
        self,
        selected_crops: dict[int, tuple[TrackCrop, ...]],
    ) -> tuple[list[TrackEmbeddingSample], int, int]:
        samples: list[TrackEmbeddingSample] = []
        cache_hits = 0
        cache_misses = 0
        for track_id, crops in sorted(selected_crops.items()):
            for crop in crops:
                metadata = {
                    "source_track_id": track_id,
                    "source_frame_index": crop.reference.frame_index,
                }
                lookup = self.cache.get(
                    image=crop.image,
                    backend_name=self.provider.backend_name,
                    model_id=self.provider.model_id,
                    inference_config=self.provider.inference_config(),
                )
                if lookup.cache_hit and lookup.embedding is not None:
                    embedding = lookup.embedding
                    cache_hits += 1
                else:
                    embedding = self.provider.embed_crop(crop.image, metadata=metadata)
                    self.cache.set(embedding, lookup.cache_key)
                    cache_misses += 1
                samples.append(
                    TrackEmbeddingSample(
                        crop_reference=crop.reference,
                        embedding=embedding,
                        quality_weight=max(crop.reference.quality_metrics.quality_score, 1e-6),
                    )
                )
        return samples, cache_hits, cache_misses

    def verify(
        self,
        *,
        source_video: str | Path,
        tracks_path: str | Path,
        semantic_memory_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> tuple[AppearanceVerificationResult, FusionResult]:
        source = Path(source_video)
        tracks = Path(tracks_path)
        semantic_path = Path(semantic_memory_path)
        if not source.is_file():
            raise AppearanceVerificationServiceError(f"Source video does not exist: {source}")
        if not tracks.is_file():
            raise AppearanceVerificationServiceError(f"MOT track file does not exist: {tracks}")
        if not semantic_path.is_file():
            raise AppearanceVerificationServiceError(
                f"Semantic memory artifact does not exist: {semantic_path}"
            )
        output_root = Path(output_dir) if output_dir is not None else self.config.output_dir
        tracks_hash_before = _sha256_file(tracks)
        source_hash_before = _sha256_file(source)
        semantic_hash_before = _sha256_file(semantic_path)
        semantic_memory = load_semantic_memory(semantic_path)
        crop_started = time.perf_counter()
        crops_by_track = self._extract_candidate_crops(
            semantic_memory=semantic_memory,
            source_video=source,
            tracks_path=tracks,
            output_root=output_root,
        )
        selected_crops = {
            track_id: select_representative_crops(crops, self.config.crop_selection)
            for track_id, crops in crops_by_track.items()
        }
        crop_seconds = time.perf_counter() - crop_started
        embedding_started = time.perf_counter()
        samples, cache_hits, cache_misses = self._embed_selected_crops(selected_crops)
        embedding_seconds = time.perf_counter() - embedding_started
        samples_by_track: dict[int, list[TrackEmbeddingSample]] = {}
        for sample in samples:
            samples_by_track.setdefault(sample.crop_reference.raw_track_id, []).append(sample)
        prototype_started = time.perf_counter()
        prototypes = []
        warnings: list[str] = []
        for track_id, track_samples in sorted(samples_by_track.items()):
            try:
                prototypes.append(
                    build_track_prototype(
                        raw_track_id=track_id,
                        samples=tuple(track_samples),
                        strategy=self.config.prototype_strategy,
                    )
                )
            except PrototypeBuildError as exc:
                warnings.append(f"track {track_id}: {exc}")
        prototype_seconds = time.perf_counter() - prototype_started
        verification_started = time.perf_counter()
        candidate_scores = score_appearance_prototypes(tuple(prototypes), self.config.verifier)
        verification_seconds = time.perf_counter() - verification_started
        appearance_result = AppearanceVerificationResult(
            query=semantic_memory.query,
            source_video=str(source),
            tracks_path=str(tracks),
            semantic_memory_reference=str(semantic_path),
            prototypes=tuple(prototypes),
            candidate_scores=candidate_scores,
            runtime_info=AppearanceRuntimeInfo(
                backend_name=self.provider.backend_name,
                model_id=self.provider.model_id,
                crop_extraction_seconds=crop_seconds,
                embedding_seconds=embedding_seconds,
                prototype_build_seconds=prototype_seconds,
                verification_seconds=verification_seconds,
                crop_count=sum(len(crops) for crops in selected_crops.values()),
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                metadata={
                    "tracks_sha256_before": tracks_hash_before,
                    "source_video_sha256_before": source_hash_before,
                    "semantic_memory_sha256_before": semantic_hash_before,
                },
            ),
            status="ok" if candidate_scores else "insufficient_appearance_evidence",
            warnings=tuple(warnings),
        )
        appearance_path = output_root / "appearance_scores.json"
        _write_json(
            output_root / "appearance_manifest.json",
            {
                "semantic_memory_reference": str(semantic_path),
                "source_video": str(source),
                "tracks_path": str(tracks),
                "selected_crops": {
                    str(track_id): [crop.reference.to_dict() for crop in crops]
                    for track_id, crops in selected_crops.items()
                },
                "prototypes": [
                    prototype.to_dict(include_vectors=self.config.include_vectors_in_json)
                    for prototype in prototypes
                ],
            },
            overwrite=self.config.overwrite,
        )
        _write_json(
            appearance_path,
            appearance_result.to_dict(include_vectors=self.config.include_vectors_in_json),
            overwrite=self.config.overwrite,
        )
        fused_scores = fuse_scores(
            semantic_memory=semantic_memory,
            appearance_scores=candidate_scores,
            config=self.config.fusion,
        )
        fusion = decide_fused_resolution(
            query=semantic_memory.query,
            fused_scores=fused_scores,
            semantic_memory_reference=str(semantic_path),
            appearance_scores_reference=str(appearance_path),
            config=self.config.fusion,
            warnings=tuple(warnings),
        )
        _write_json(
            output_root / "fusion_result.json", fusion.to_dict(), overwrite=self.config.overwrite
        )
        tracks_hash_after = _sha256_file(tracks)
        source_hash_after = _sha256_file(source)
        semantic_hash_after = _sha256_file(semantic_path)
        if tracks_hash_before != tracks_hash_after:
            raise AppearanceVerificationServiceError("MOT artifact changed during M4 verification.")
        if source_hash_before != source_hash_after:
            raise AppearanceVerificationServiceError("Source video changed during M4 verification.")
        if semantic_hash_before != semantic_hash_after:
            raise AppearanceVerificationServiceError(
                "M3 semantic memory changed during M4 verification."
            )
        return appearance_result, fusion

    def close(self) -> None:
        self.provider.close()
