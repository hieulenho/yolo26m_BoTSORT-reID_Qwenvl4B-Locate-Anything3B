"""Checks that language ablation variants reuse the same benchmark inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from football_tracking.locate_tracking.benchmark.manifest import load_prediction_manifest
from football_tracking.locate_tracking.experiments.schemas import LanguageAblationVariant


@dataclass(frozen=True)
class ArtifactReuseReport:
    compatible: bool
    query_keys_by_variant: dict[str, list[str]]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "query_keys_by_variant": self.query_keys_by_variant,
            "warnings": list(self.warnings),
        }


def check_artifact_reuse(variants: tuple[LanguageAblationVariant, ...]) -> ArtifactReuseReport:
    keys_by_variant: dict[str, list[str]] = {}
    warnings: list[str] = []
    reference: set[str] | None = None
    for variant in variants:
        manifest = load_prediction_manifest(variant.prediction_manifest)
        keys = {
            f"{prediction.sequence_name}:{prediction.query_id}"
            for prediction in manifest.predictions
        }
        keys_by_variant[variant.variant_id] = sorted(keys)
        if reference is None:
            reference = keys
        elif keys != reference:
            warnings.append(f"{variant.variant_id} has a different query set.")
    return ArtifactReuseReport(
        compatible=not warnings,
        query_keys_by_variant=keys_by_variant,
        warnings=tuple(warnings),
    )
