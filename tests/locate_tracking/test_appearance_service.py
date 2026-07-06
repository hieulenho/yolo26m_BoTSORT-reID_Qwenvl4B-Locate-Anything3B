from __future__ import annotations

import hashlib
from pathlib import Path

from football_tracking.locate_tracking.appearance.mock_backend import (
    MockAppearanceEmbeddingProvider,
)
from football_tracking.locate_tracking.appearance.service import AppearanceVerificationService
from tests.locate_tracking.appearance_test_utils import (
    appearance_config,
    semantic_memory_fixture,
    tiny_tracks,
    tiny_video,
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_appearance_service_end_to_end_with_mock_backend_and_immutability(tmp_path: Path) -> None:
    video = tiny_video(tmp_path / "source.avi")
    tracks = tiny_tracks(tmp_path / "tracks.txt")
    semantic = semantic_memory_fixture(tmp_path / "semantic_memory.json")
    before = (_hash(video), _hash(tracks), _hash(semantic))
    provider = MockAppearanceEmbeddingProvider(
        {
            (7, 1): (1.0, 0.0, 0.0),
            (7, 6): (0.99, 0.01, 0.0),
            (11, 1): (1.0, 0.0, 0.0),
            (11, 6): (0.0, 1.0, 0.0),
        }
    )
    service = AppearanceVerificationService(
        config=appearance_config(tmp_path),
        provider=provider,
    )

    appearance, fusion = service.verify(
        source_video=video,
        tracks_path=tracks,
        semantic_memory_path=semantic,
        output_dir=tmp_path / "appearance",
    )

    assert appearance.status == "ok"
    assert fusion.status == "resolved"
    assert fusion.selected_track_id == 7
    assert (tmp_path / "appearance" / "appearance_manifest.json").is_file()
    assert (tmp_path / "appearance" / "appearance_scores.json").is_file()
    assert (tmp_path / "appearance" / "fusion_result.json").is_file()
    assert before == (_hash(video), _hash(tracks), _hash(semantic))


def test_appearance_service_cache_reuse(tmp_path: Path) -> None:
    video = tiny_video(tmp_path / "source.avi")
    tracks = tiny_tracks(tmp_path / "tracks.txt")
    semantic = semantic_memory_fixture(tmp_path / "semantic_memory.json")
    provider = MockAppearanceEmbeddingProvider(
        {
            (7, 1): (1.0, 0.0, 0.0),
            (7, 6): (0.99, 0.01, 0.0),
            (11, 1): (1.0, 0.0, 0.0),
            (11, 6): (0.0, 1.0, 0.0),
        }
    )
    config = appearance_config(tmp_path)
    service = AppearanceVerificationService(config=config, provider=provider)

    service.verify(
        source_video=video,
        tracks_path=tracks,
        semantic_memory_path=semantic,
        output_dir=tmp_path / "run1",
    )
    first_call_count = provider.call_count
    service.verify(
        source_video=video,
        tracks_path=tracks,
        semantic_memory_path=semantic,
        output_dir=tmp_path / "run2",
    )

    assert first_call_count == 4
    assert provider.call_count == 4


def test_appearance_service_one_candidate_without_evidence_uses_missing_policy(
    tmp_path: Path,
) -> None:
    video = tiny_video(tmp_path / "source.avi")
    tracks = tiny_tracks(tmp_path / "tracks.txt")
    semantic = semantic_memory_fixture(tmp_path / "semantic_memory.json")
    provider = MockAppearanceEmbeddingProvider({(7, 1): (1.0, 0.0), (7, 6): (1.0, 0.0)})
    config = appearance_config(tmp_path)
    service = AppearanceVerificationService(config=config, provider=provider)

    _, fusion = service.verify(
        source_video=video,
        tracks_path=tracks,
        semantic_memory_path=semantic,
        output_dir=tmp_path / "appearance",
    )

    assert fusion.candidate_scores
    assert any(score.appearance_status in {"verified", "weak"} for score in fusion.candidate_scores)
