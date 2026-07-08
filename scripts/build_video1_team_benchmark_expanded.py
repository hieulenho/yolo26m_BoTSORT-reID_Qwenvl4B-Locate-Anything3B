from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path("data/team_benchmark/video_1")
BENCHMARK_NAME = "video_1_team_language_expanded"
SEQUENCE_NAME = "video_1"
SOURCE_VIDEO = "F:/videos/1.mp4"
TRACKS_PATH = "F:/videos/1_Tracking_qwen.txt"
FRAME_COUNT = 1399
CONTACT_SHEET_DIR = "outputs/team_benchmark/video_1_annotation/contact_sheets"
TRACK19_RESOLUTION = (
    "outputs/locate_tracking/runs/video_1_locateanything_track19_window/"
    "final_resolution.json"
)
ANNOTATION_CSV = "data/team_benchmark/video_1/track_annotation_expanded.csv"
PIPELINE_A_PREDICTIONS = "pipeline_a_yolo26m_botsort_reid_qwen4b_expanded_bootstrap.json"
PIPELINE_C_PREDICTIONS = (
    "pipeline_c_yolo26m_botsort_reid_locateanything3b_qwen4b_expanded_bootstrap.json"
)


ANNOTATIONS: list[dict[str, Any]] = [
    {
        "track_id": 7,
        "team_label": "light_blue",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 1399,
        "notes": "Contact sheet verified; stable light-blue player.",
    },
    {
        "track_id": 31,
        "team_label": "light_blue",
        "role_label": "player",
        "start_frame": 345,
        "end_frame": 1351,
        "notes": "Contact sheet verified; light-blue player.",
    },
    {
        "track_id": 37,
        "team_label": "light_blue",
        "role_label": "player",
        "start_frame": 428,
        "end_frame": 1356,
        "notes": "Contact sheet verified; light-blue player.",
    },
    {
        "track_id": 10,
        "team_label": "light_blue",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 558,
        "notes": "Contact sheet verified in earlier stable window; late sample partially occluded.",
    },
    {
        "track_id": 45,
        "team_label": "light_blue",
        "role_label": "player",
        "start_frame": 617,
        "end_frame": 1399,
        "notes": "Contact sheet verified; stable light-blue player.",
    },
    {
        "track_id": 48,
        "team_label": "light_blue",
        "role_label": "player",
        "start_frame": 620,
        "end_frame": 1269,
        "notes": "Contact sheet verified; light-blue player with crowded late frame.",
    },
    {
        "track_id": 49,
        "team_label": "light_blue",
        "role_label": "player",
        "start_frame": 668,
        "end_frame": 1222,
        "notes": "Contact sheet verified; light-blue player.",
    },
    {
        "track_id": 43,
        "team_label": "light_blue",
        "role_label": "player",
        "start_frame": 597,
        "end_frame": 1268,
        "notes": "Contact sheet verified; light-blue player.",
    },
    {
        "track_id": 3,
        "team_label": "light_blue",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 405,
        "notes": "Contact sheet verified; light-blue player.",
    },
    {
        "track_id": 8,
        "team_label": "light_blue",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 400,
        "notes": "Contact sheet verified; light-blue player.",
    },
    {
        "track_id": 11,
        "team_label": "light_blue",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 402,
        "notes": "Contact sheet verified; light-blue player.",
    },
    {
        "track_id": 51,
        "team_label": "dark_blue",
        "role_label": "player",
        "start_frame": 696,
        "end_frame": 1399,
        "notes": "Contact sheet verified; stable dark-blue player.",
    },
    {
        "track_id": 12,
        "team_label": "dark_blue",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 698,
        "notes": "Contact sheet verified; dark-blue player.",
    },
    {
        "track_id": 32,
        "team_label": "dark_blue",
        "role_label": "player",
        "start_frame": 348,
        "end_frame": 915,
        "notes": "Contact sheet verified; dark-blue player.",
    },
    {
        "track_id": 50,
        "team_label": "dark_blue",
        "role_label": "player",
        "start_frame": 683,
        "end_frame": 1254,
        "notes": "Contact sheet verified; dark-blue player.",
    },
    {
        "track_id": 18,
        "team_label": "dark_blue",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 502,
        "notes": "Contact sheet verified; dark-blue player.",
    },
    {
        "track_id": 2,
        "team_label": "dark_blue",
        "role_label": "player",
        "start_frame": 5,
        "end_frame": 445,
        "notes": "Contact sheet verified; dark-blue player.",
    },
    {
        "track_id": 38,
        "team_label": "dark_blue",
        "role_label": "player",
        "start_frame": 522,
        "end_frame": 946,
        "notes": "Contact sheet verified; dark-blue player.",
    },
    {
        "track_id": 46,
        "team_label": "dark_blue",
        "role_label": "player",
        "start_frame": 617,
        "end_frame": 1064,
        "notes": "Contact sheet verified; dark-blue player.",
    },
    {
        "track_id": 4,
        "team_label": "goalkeeper_red",
        "role_label": "goalkeeper",
        "start_frame": 5,
        "end_frame": 392,
        "notes": "Contact sheet verified; red goalkeeper. Later frame is crowded/occluded.",
    },
    {
        "track_id": 19,
        "team_label": "goalkeeper_green",
        "role_label": "goalkeeper",
        "start_frame": 40,
        "end_frame": 320,
        "notes": "Verified by LocateAnything track19 window and contact-sheet review.",
    },
]


QUERIES: list[dict[str, Any]] = [
    {
        "query_id": "q_all_light_blue_players",
        "query_text": "all annotated players wearing the light blue kit",
        "expected_team_label": "light_blue",
        "expected_track_ids": [3, 7, 8, 10, 11, 31, 37, 43, 45, 48, 49],
        "start_frame": 5,
        "end_frame": 1399,
        "difficulty": "medium",
        "notes": "Team-level query over manually verified light-blue tracks.",
    },
    {
        "query_id": "q_all_dark_blue_players",
        "query_text": "all annotated players wearing the dark blue kit",
        "expected_team_label": "dark_blue",
        "expected_track_ids": [2, 12, 18, 32, 38, 46, 50, 51],
        "start_frame": 5,
        "end_frame": 1399,
        "difficulty": "medium",
        "notes": "Team-level query over manually verified dark-blue tracks.",
    },
    {
        "query_id": "q_goalkeeper_red",
        "query_text": "the goalkeeper wearing red",
        "expected_team_label": "goalkeeper_red",
        "expected_track_ids": [4],
        "start_frame": 5,
        "end_frame": 392,
        "difficulty": "easy",
        "notes": "Single-target goalkeeper query.",
    },
    {
        "query_id": "q_goalkeeper_green",
        "query_text": "the goalkeeper wearing green",
        "expected_team_label": "goalkeeper_green",
        "expected_track_ids": [19],
        "start_frame": 40,
        "end_frame": 320,
        "difficulty": "easy",
        "notes": "Single-target goalkeeper query resolved by previous LocateAnything run.",
    },
    {
        "query_id": "q_light_blue_long_track",
        "query_text": "the long-lived light blue player track visible throughout the clip",
        "expected_team_label": "light_blue",
        "expected_track_ids": [7],
        "start_frame": 5,
        "end_frame": 1399,
        "difficulty": "hard",
        "notes": "Identity-specific query for the longest stable light-blue track.",
    },
    {
        "query_id": "q_dark_blue_penalty_area_group",
        "query_text": "the dark blue player cluster near the penalty area",
        "expected_team_label": "dark_blue",
        "expected_track_ids": [50, 51],
        "start_frame": 683,
        "end_frame": 1399,
        "difficulty": "hard",
        "notes": "Multi-target query using dark-blue tracks visible near the penalty area.",
    },
]


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def evidence_frames(annotation: dict[str, Any], max_frames: int = 4) -> list[int]:
    start = int(annotation["start_frame"])
    end = int(annotation["end_frame"])
    if start == end:
        return [start]
    if max_frames <= 1:
        return [(start + end) // 2]
    return sorted({round(start + i * (end - start) / (max_frames - 1)) for i in range(max_frames)})


def make_track_prediction(annotation: dict[str, Any], pipeline: str) -> dict[str, Any]:
    is_locate = pipeline == "locate"
    confidence = 0.88 if is_locate else 0.82
    if annotation["track_id"] == 19 and is_locate:
        confidence = 0.85
    return {
        "sequence_name": SEQUENCE_NAME,
        "track_id": annotation["track_id"],
        "status": "resolved",
        "team_label": annotation["team_label"],
        "role_label": annotation["role_label"],
        "confidence": confidence,
        "evidence_frames": evidence_frames(annotation),
        "metadata": {
            "source": (
                "manual_contact_sheet_bootstrap_for_expanded_benchmark"
                if annotation["track_id"] != 19
                else TRACK19_RESOLUTION
            ),
            "contact_sheet": f"{CONTACT_SHEET_DIR}/track_{int(annotation['track_id']):04d}.jpg",
            "not_model_claim": annotation["track_id"] != 19,
        },
    }


def make_query_prediction(query: dict[str, Any], pipeline: str) -> dict[str, Any]:
    is_locate = pipeline == "locate"
    track_count = len(query["expected_track_ids"])
    return {
        "sequence_name": SEQUENCE_NAME,
        "query_id": query["query_id"],
        "status": "resolved",
        "selected_track_ids": query["expected_track_ids"],
        "team_label": query["expected_team_label"],
        "confidence": 0.86 if is_locate else 0.8,
        "support_ratio": 0.88 if is_locate else 0.74,
        "grounding_call_count": max(4, track_count * 2) if is_locate else 0,
        "runtime_seconds": None,
        "metadata": {
            "source": "bootstrap_prediction_for_evaluator_validation",
            "not_model_claim": True,
        },
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "benchmark_name": BENCHMARK_NAME,
        "benchmark_version": "0.2.0",
        "dataset_name": "custom_video_1_sportsmot_style",
        "split": "dev",
        "annotation_policy": (
            "Expanded real-video benchmark. Team labels are manually verified from contact sheets; "
            "tracks with visible identity switches are excluded."
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
                    "excluded_track_ids": [1, 13, 28, 39],
                    "excluded_reason": (
                        "Contact-sheet review showed identity switch, referee/non-player, "
                        "or poor visibility."
                    ),
                },
            }
        ],
        "metadata": {
            "manual_annotation_template": ANNOTATION_CSV,
            "contact_sheet_dir": CONTACT_SHEET_DIR,
            "scope": (
                "bootstrap expanded benchmark; replace bootstrap predictions with real "
                "model outputs for final claims"
            ),
        },
    }

    write_json(OUTPUT_DIR / "benchmark_manifest_expanded.json", manifest)
    annotation_csv_path = OUTPUT_DIR / "track_annotation_expanded.csv"
    with annotation_csv_path.open("w", encoding="utf-8", newline="") as handle:
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

    for pipeline, variant_id, variant_name, pipeline_type in [
        (
            "qwen",
            "pipeline_a_yolo26m_botsort_reid_qwen4b_video_1_expanded_bootstrap",
            "Pipeline A - YOLO26m + BoT-SORT ReID + Qwen3-VL 4B video_1 expanded bootstrap",
            "yolo_botsort_qwen",
        ),
        (
            "locate",
            "pipeline_c_yolo26m_botsort_reid_locateanything3b_qwen4b_video_1_expanded_bootstrap",
            (
                "Pipeline C - YOLO26m + BoT-SORT ReID + LocateAnything 3B + "
                "Qwen3-VL 4B video_1 expanded bootstrap"
            ),
            "yolo_botsort_locateanything_qwen",
        ),
    ]:
        prediction = {
            "variant_id": variant_id,
            "variant_name": variant_name,
            "benchmark_name": BENCHMARK_NAME,
            "pipeline_type": pipeline_type,
            "created_at": now(),
            "track_predictions": [
                make_track_prediction(annotation, pipeline) for annotation in ANNOTATIONS
            ],
            "query_predictions": [make_query_prediction(query, pipeline) for query in QUERIES],
            "metadata": {
                "bootstrap": True,
                "warning": (
                    "These expanded predictions validate the benchmark plumbing and use "
                    "contact-sheet labels. They are not a full model-run claim except "
                    "for the previously resolved track 19 evidence."
                ),
            },
        }
        suffix = PIPELINE_A_PREDICTIONS if pipeline == "qwen" else PIPELINE_C_PREDICTIONS
        write_json(OUTPUT_DIR / suffix, prediction)

    readme_lines = [
        "# Video 1 Expanded Team Benchmark",
        "",
        "This folder contains an expanded real-video benchmark for `F:/videos/1.mp4`.",
        "",
        "- `benchmark_manifest_expanded.json`: 21 reviewed tracks and 6 language queries.",
        "- `track_annotation_expanded.csv`: CSV view of labels and contact-sheet paths.",
        (
            "- `pipeline_a_yolo26m_botsort_reid_qwen4b_expanded_bootstrap.json`: "
            "Pipeline A bootstrap predictions."
        ),
        (
            "- `pipeline_c_yolo26m_botsort_reid_locateanything3b_qwen4b_expanded_bootstrap.json`: "
            "Pipeline C bootstrap predictions."
        ),
        "- Pipeline B requires a true LocateAnything-only prediction manifest.",
        "",
        "Important: the expanded prediction manifests are bootstrap artifacts based on",
        "contact-sheet labels. They are useful for checking benchmark mechanics over more",
        "tracks, but final research claims should replace them with true model outputs.",
        "",
    ]
    readme = "\n".join(readme_lines)
    (OUTPUT_DIR / "README_expanded.md").write_text(readme, encoding="utf-8")
    print(f"Wrote expanded benchmark files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
