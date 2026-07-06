from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path("data/team_benchmark/video_2")
BENCHMARK_NAME = "video_2_team_language_expanded"
SEQUENCE_NAME = "video_2"
SOURCE_VIDEO = "F:/videos/2.mp4"
TRACKS_PATH = "F:/videos/2_Tracking_qwen.txt"
FRAME_COUNT = 855
CONTACT_SHEET_DIR = "outputs/team_benchmark/video_2_annotation/contact_sheets_large"
ANNOTATION_CSV = "data/team_benchmark/video_2/track_annotation_expanded.csv"


ANNOTATIONS: list[dict[str, Any]] = [
    {
        "track_id": 6,
        "team_label": "dark_kit",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 855,
        "notes": "Contact sheet verified; stable dark-kit player.",
    },
    {
        "track_id": 1,
        "team_label": "dark_kit",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 854,
        "notes": "Contact sheet verified; dark-kit player, late crowded frame.",
    },
    {
        "track_id": 15,
        "team_label": "dark_kit",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 852,
        "notes": "Contact sheet verified; dark-kit player.",
    },
    {
        "track_id": 14,
        "team_label": "dark_kit",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 855,
        "notes": "Contact sheet verified; dark-kit player.",
    },
    {
        "track_id": 4,
        "team_label": "dark_kit",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 855,
        "notes": "Contact sheet verified; dark-kit player.",
    },
    {
        "track_id": 5,
        "team_label": "dark_kit",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 855,
        "notes": "Contact sheet verified; dark-kit player.",
    },
    {
        "track_id": 12,
        "team_label": "dark_kit",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 637,
        "notes": "Contact sheet verified; dark-kit player, later crowded frame excluded.",
    },
    {
        "track_id": 22,
        "team_label": "dark_kit",
        "role_label": "player",
        "start_frame": 15,
        "end_frame": 620,
        "notes": "Contact sheet verified; dark-kit player.",
    },
    {
        "track_id": 16,
        "team_label": "dark_kit",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 554,
        "notes": "Contact sheet verified; dark-kit player.",
    },
    {
        "track_id": 13,
        "team_label": "yellow_kit",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 855,
        "notes": "Contact sheet verified; stable yellow-kit player.",
    },
    {
        "track_id": 7,
        "team_label": "yellow_kit",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 855,
        "notes": "Contact sheet verified; yellow-kit player.",
    },
    {
        "track_id": 20,
        "team_label": "yellow_kit",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 855,
        "notes": "Contact sheet verified; yellow-kit player.",
    },
    {
        "track_id": 3,
        "team_label": "yellow_kit",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 855,
        "notes": "Contact sheet verified; yellow-kit player.",
    },
    {
        "track_id": 19,
        "team_label": "yellow_kit",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 662,
        "notes": "Contact sheet verified; yellow-kit player.",
    },
    {
        "track_id": 18,
        "team_label": "yellow_kit",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 819,
        "notes": "Contact sheet verified; yellow-kit player.",
    },
    {
        "track_id": 10,
        "team_label": "yellow_kit",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 613,
        "notes": "Contact sheet verified; yellow-kit player.",
    },
    {
        "track_id": 21,
        "team_label": "yellow_kit",
        "role_label": "player",
        "start_frame": 10,
        "end_frame": 811,
        "notes": "Contact sheet verified; yellow-kit player.",
    },
    {
        "track_id": 28,
        "team_label": "yellow_kit",
        "role_label": "player",
        "start_frame": 263,
        "end_frame": 679,
        "notes": "Contact sheet verified; yellow-kit player.",
    },
    {
        "track_id": 11,
        "team_label": "yellow_kit",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 352,
        "notes": "Contact sheet verified; yellow-kit player.",
    },
    {
        "track_id": 17,
        "team_label": "yellow_kit",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 297,
        "notes": "Contact sheet verified; yellow-kit player, moderate crowding.",
    },
    {
        "track_id": 31,
        "team_label": "yellow_kit",
        "role_label": "player",
        "start_frame": 326,
        "end_frame": 813,
        "notes": "Contact sheet verified; yellow-kit player, moderate crowding.",
    },
    {
        "track_id": 8,
        "team_label": "goalkeeper_orange",
        "role_label": "goalkeeper",
        "start_frame": 215,
        "end_frame": 855,
        "notes": "Contact sheet verified; orange goalkeeper.",
    },
]


QUERIES: list[dict[str, Any]] = [
    {
        "query_id": "q_all_dark_kit_players",
        "query_text": "all annotated players wearing the dark kit",
        "expected_team_label": "dark_kit",
        "expected_track_ids": [1, 4, 5, 6, 12, 14, 15, 16, 22],
        "start_frame": 5,
        "end_frame": 855,
        "difficulty": "medium",
        "notes": "Team-level query over manually verified dark-kit tracks.",
    },
    {
        "query_id": "q_all_yellow_kit_players",
        "query_text": "all annotated players wearing the yellow kit",
        "expected_team_label": "yellow_kit",
        "expected_track_ids": [3, 7, 10, 11, 13, 17, 18, 19, 20, 21, 28, 31],
        "start_frame": 5,
        "end_frame": 855,
        "difficulty": "medium",
        "notes": "Team-level query over manually verified yellow-kit tracks.",
    },
    {
        "query_id": "q_goalkeeper_orange",
        "query_text": "the goalkeeper wearing orange",
        "expected_team_label": "goalkeeper_orange",
        "expected_track_ids": [8],
        "start_frame": 215,
        "end_frame": 855,
        "difficulty": "easy",
        "notes": "Single-target goalkeeper query.",
    },
]


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "benchmark_name": BENCHMARK_NAME,
        "benchmark_version": "0.2.0",
        "dataset_name": "custom_video_2_sportsmot_style",
        "split": "dev",
        "annotation_policy": (
            "Expanded real-video benchmark. Team labels are manually verified from "
            "large contact sheets; visibly switched tracks are excluded."
        ),
        "created_at": now(),
        "sequences": [
            {
                "sequence_name": SEQUENCE_NAME,
                "split": "dev",
                "source_video": SOURCE_VIDEO,
                "tracks_path": TRACKS_PATH,
                "mot_ground_truth_path": None,
                "frame_count": FRAME_COUNT,
                "track_annotations": ANNOTATIONS,
                "query_annotations": QUERIES,
                "metadata": {
                    "contact_sheet_index": f"{CONTACT_SHEET_DIR}/index.csv",
                    "excluded_track_ids": [25],
                    "excluded_reason": "Contact-sheet review suggested possible identity switch.",
                },
            }
        ],
        "metadata": {
            "manual_annotation_template": ANNOTATION_CSV,
            "contact_sheet_dir": CONTACT_SHEET_DIR,
            "scope": "video_2 manually verified team benchmark",
        },
    }
    write_json(OUTPUT_DIR / "benchmark_manifest_expanded.json", manifest)
    with (OUTPUT_DIR / "track_annotation_expanded.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = [
            "sequence_name",
            "track_id",
            "start_frame",
            "end_frame",
            "team_label",
            "role_label",
            "notes",
            "contact_sheet",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for annotation in ANNOTATIONS:
            writer.writerow(
                {
                    "sequence_name": SEQUENCE_NAME,
                    "track_id": annotation["track_id"],
                    "start_frame": annotation["start_frame"],
                    "end_frame": annotation["end_frame"],
                    "team_label": annotation["team_label"],
                    "role_label": annotation["role_label"],
                    "notes": annotation["notes"],
                    "contact_sheet": (
                        f"{CONTACT_SHEET_DIR}/track_{int(annotation['track_id']):04d}.jpg"
                    ),
                }
            )
    readme = "\n".join(
        [
            "# Video 2 Expanded Team Benchmark",
            "",
            "Manual team-label benchmark for `F:/videos/2.mp4`.",
            "",
            "- `benchmark_manifest_expanded.json`: 22 reviewed tracks and 3 queries.",
            "- `track_annotation_expanded.csv`: label table with contact-sheet references.",
            "",
        ]
    )
    (OUTPUT_DIR / "README_expanded.md").write_text(readme, encoding="utf-8")
    print(f"Wrote video_2 benchmark files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
