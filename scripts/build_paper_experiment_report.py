"""Build paper-style experiment tables from saved project artifacts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "reports" / "paper_experiments"


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def detector_metrics_from_baseline(path: Path) -> dict[str, Any] | None:
    payload = load_json(path)
    if not isinstance(payload, dict):
        return None
    model = payload.get("model", {})
    metrics = payload.get("metrics", {})
    inference = payload.get("inference", {})
    return {
        "weights": model.get("weights"),
        "image_size": inference.get("imgsz"),
        "precision": metric(metrics.get("precision")),
        "recall": metric(metrics.get("recall")),
        "map50": metric(metrics.get("map50")),
        "map50_95": metric(metrics.get("map50_95")),
        "map75": metric(metrics.get("map75")),
        "fps": None,
    }


def detector_metrics_from_finetuned(path: Path) -> dict[str, Any] | None:
    payload = load_json(path)
    if not isinstance(payload, dict):
        return None
    return {
        "weights": payload.get("checkpoint"),
        "image_size": payload.get("image_size"),
        "precision": metric(payload.get("precision")),
        "recall": metric(payload.get("recall")),
        "map50": metric(payload.get("map50")),
        "map50_95": metric(payload.get("map50_95")),
        "map75": metric(payload.get("map75")),
        "fps": None,
    }


def build_detector_rows() -> list[dict[str, Any]]:
    specs = [
        (
            "YOLOv8m pretrained",
            "COCO pretrain only",
            "baseline",
            ROOT
            / "outputs"
            / "metrics"
            / "detector_baselines"
            / "yolov8m_pretrained_sportsmot_val"
            / "yolov8m_pretrained_baseline.json",
            detector_metrics_from_baseline,
        ),
        (
            "YOLO26n pretrained",
            "COCO pretrain only",
            "small model-size baseline",
            ROOT
            / "outputs"
            / "metrics"
            / "detector_baselines"
            / "yolo26n_pretrained_sportsmot_val"
            / "yolov8m_pretrained_baseline.json",
            detector_metrics_from_baseline,
        ),
        (
            "YOLO26m pretrained",
            "COCO pretrain only",
            "same family before fine-tune",
            ROOT
            / "outputs"
            / "metrics"
            / "detector_baselines"
            / "yolo26m_pretrained_sportsmot_val"
            / "yolov8m_pretrained_baseline.json",
            detector_metrics_from_baseline,
        ),
        (
            "YOLO26m fine-tuned",
            "SportsMOT train",
            "method",
            ROOT / "outputs" / "metrics" / "football" / "yolo26m" / "yolo26m_val.json",
            detector_metrics_from_finetuned,
        ),
    ]
    rows: list[dict[str, Any]] = []
    for detector, train_data, note, path, reader in specs:
        metrics = reader(path)
        row = {
            "detector": detector,
            "train_data": train_data,
            "eval_data": "SportsMOT football val",
            "metric_path": str(path.relative_to(ROOT)),
            "note": note,
            "status": "ok" if metrics else "missing",
        }
        if metrics:
            row.update(metrics)
        rows.append(row)
    return rows


def load_tracker_rows(path: Path, labels: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        return []
    rows = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("tracker"))
        meta = labels.get(name, {"label": name, "reid": "unknown", "note": ""})
        rows.append(
            {
                "tracker": meta["label"],
                "raw_tracker": name,
                "reid": meta["reid"],
                "sequence_count": item.get("sequence_count"),
                "frame_count": item.get("frame_count"),
                "HOTA": metric(item.get("HOTA"), 3),
                "DetA": metric(item.get("DetA"), 3),
                "AssA": metric(item.get("AssA"), 3),
                "MOTA": metric(item.get("MOTA"), 3),
                "IDF1": metric(item.get("IDF1"), 3),
                "IDSW": item.get("IDSW"),
                "Frag": item.get("Frag"),
                "FP": item.get("FP"),
                "FN": item.get("FN"),
                "tracker_fps": metric(item.get("tracker_fps"), 3),
                "cached_pipeline_fps": metric(item.get("cached_pipeline_fps"), 3),
                "unique_predicted_ids": item.get("unique_predicted_ids"),
                "note": meta["note"],
                "metric_path": str(path.relative_to(ROOT)),
            }
        )
    return rows


def build_tracking_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows += load_tracker_rows(
        ROOT
        / "outputs"
        / "metrics"
        / "experiments"
        / "yolo26m_botsort_all"
        / "sort_vs_deepsort_overall.json",
        {
            "sort": {"label": "SORT", "reid": "No", "note": "classic Kalman/IoU baseline"},
            "deepsort": {"label": "DeepSORT", "reid": "Yes", "note": "appearance baseline"},
            "botsort_reid": {
                "label": "BoT-SORT ReID balanced",
                "reid": "Yes",
                "note": "earlier balanced preset",
            },
        },
    )
    rows += load_tracker_rows(
        ROOT
        / "outputs"
        / "metrics"
        / "experiments"
        / "yolo26m_bytetrack_all"
        / "tracker_summary.json",
        {
            "bytetrack": {
                "label": "ByteTrack",
                "reid": "No",
                "note": "fast association baseline",
            }
        },
    )
    rows += load_tracker_rows(
        ROOT
        / "outputs"
        / "metrics"
        / "experiments"
        / "yolo26m_botsort_no_reid_all"
        / "tracker_summary.json",
        {
            "botsort": {
                "label": "BoT-SORT no ReID",
                "reid": "No",
                "note": "same BoT-SORT thresholds, ReID disabled",
            }
        },
    )
    rows += load_tracker_rows(
        ROOT
        / "outputs"
        / "metrics"
        / "experiments"
        / "yolo26m_botsort_identity_stable_all"
        / "sort_vs_deepsort_overall.json",
        {
            "botsort_reid": {
                "label": "BoT-SORT ReID identity-stable",
                "reid": "Yes",
                "note": "current recommended tracker preset",
            }
        },
    )
    order = {
        "SORT": 0,
        "ByteTrack": 1,
        "DeepSORT": 2,
        "BoT-SORT no ReID": 3,
        "BoT-SORT ReID balanced": 4,
        "BoT-SORT ReID identity-stable": 5,
    }
    return sorted(rows, key=lambda row: order.get(str(row["tracker"]), 99))


def infer_vlm_settings(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    keyframes = None
    max_tokens = None
    match = re.search(r"(\d+)kf_(\d+)", name)
    if match:
        keyframes = int(match.group(1))
        max_tokens = int(match.group(2))
    if name.endswith("compact"):
        keyframes = keyframes or payload.get("image_count")
        max_tokens = max_tokens or 768
    return {"keyframes": keyframes, "max_tokens": max_tokens}


def collect_context_track_ids(run_dir: Path) -> set[int]:
    candidates = [run_dir / "vlm_context.json", run_dir / "context.json"]
    text = ""
    for path in candidates:
        if path.exists():
            text += path.read_text(encoding="utf-8", errors="ignore")
    return {int(value) for value in re.findall(r'"track_id"\s*:\s*(\d+)', text)}


def vlm_track_ref_audit(run_dir: Path, answer: str) -> dict[str, Any]:
    refs = [int(value) for value in re.findall(r"(?:track\s*)?ID\s*#?\s*(\d+)", answer, re.I)]
    context_ids = collect_context_track_ids(run_dir)
    unsupported = [value for value in refs if context_ids and value not in context_ids]
    grounded = [value for value in refs if not context_ids or value in context_ids]
    hallucination_rate = None
    if refs:
        hallucination_rate = round(len(unsupported) / len(refs), 4)
    return {
        "track_ref_count": len(refs),
        "grounded_track_refs": len(grounded),
        "unsupported_track_refs": len(unsupported),
        "track_ref_hallucination_rate": hallucination_rate,
    }


def build_vlm_rows() -> list[dict[str, Any]]:
    video_root = Path("F:/videos")
    specs = [
        ("2kf_256", video_root / "1_vlm_2kf_256"),
        ("2kf_512", video_root / "1_vlm_2kf_512"),
        ("2kf_768", video_root / "1_vlm_2kf_768"),
        ("3kf_512", video_root / "1_vlm_3kf_512"),
        ("compact", video_root / "1_vlm_tracking_report_compact"),
        ("video2_default", video_root / "2_vlm"),
    ]
    rows: list[dict[str, Any]] = []
    for run_name, run_dir in specs:
        answer_json = run_dir / "vlm_answer.json"
        payload = load_json(answer_json)
        if not isinstance(payload, dict):
            rows.append(
                {
                    "run": run_name,
                    "status": "missing",
                    "run_dir": str(run_dir),
                    "answer_json": str(answer_json),
                }
            )
            continue
        answer = str(payload.get("answer", ""))
        settings = infer_vlm_settings(run_dir.name, payload)
        error = str(payload.get("error") or payload.get("model_result", {}).get("error") or "")
        audit = vlm_track_ref_audit(run_dir, answer)
        rows.append(
            {
                "run": run_name,
                "status": payload.get("status"),
                "model_id": payload.get("model_id") or payload.get("model", {}).get("model_id"),
                "image_count": payload.get("image_count") or payload.get("summary", {}).get("keyframe_count"),
                "keyframes": settings["keyframes"],
                "max_tokens": settings["max_tokens"],
                "answer_chars": len(answer),
                "oom": "yes" if "outofmemory" in error.lower() or "cuda out of memory" in error.lower() else "no",
                "error": error[:220],
                **audit,
                "run_dir": str(run_dir),
                "answer_json": str(answer_json),
                "note": "heuristic track-reference audit; human factual audit still recommended",
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join(["---"] * len(fields)) + " |"
    body = []
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            values.append("" if value is None else str(value).replace("\n", " "))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def write_report(
    detector_rows: list[dict[str, Any]],
    tracking_rows: list[dict[str, Any]],
    vlm_rows: list[dict[str, Any]],
) -> Path:
    lines = [
        "# Paper Experiment Matrix",
        "",
        "Generated from saved artifacts in this repository. Detector FPS is not recorded by the current evaluator, so those cells are intentionally empty until a dedicated detector timing benchmark is run.",
        "",
        "## Detector Baselines",
        "",
        markdown_table(
            detector_rows,
            ["detector", "train_data", "map50", "map50_95", "recall", "precision", "fps", "note"],
        ),
        "",
        "## Tracking Baselines",
        "",
        markdown_table(
            tracking_rows,
            [
                "tracker",
                "reid",
                "HOTA",
                "DetA",
                "AssA",
                "MOTA",
                "IDF1",
                "IDSW",
                "tracker_fps",
                "note",
            ],
        ),
        "",
        "## Qwen VLM Runs",
        "",
        markdown_table(
            vlm_rows,
            [
                "run",
                "keyframes",
                "max_tokens",
                "status",
                "oom",
                "image_count",
                "answer_chars",
                "grounded_track_refs",
                "track_ref_hallucination_rate",
                "note",
            ],
        ),
        "",
    ]
    path = OUT_DIR / "experiment_matrix.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    detector_rows = build_detector_rows()
    tracking_rows = build_tracking_rows()
    vlm_rows = build_vlm_rows()

    write_json(OUT_DIR / "detector_baselines.json", detector_rows)
    write_csv(
        OUT_DIR / "detector_baselines.csv",
        detector_rows,
        [
            "detector",
            "train_data",
            "eval_data",
            "weights",
            "image_size",
            "precision",
            "recall",
            "map50",
            "map50_95",
            "map75",
            "fps",
            "note",
            "status",
            "metric_path",
        ],
    )
    write_json(OUT_DIR / "tracking_baselines.json", tracking_rows)
    write_csv(
        OUT_DIR / "tracking_baselines.csv",
        tracking_rows,
        [
            "tracker",
            "raw_tracker",
            "reid",
            "sequence_count",
            "frame_count",
            "HOTA",
            "DetA",
            "AssA",
            "MOTA",
            "IDF1",
            "IDSW",
            "Frag",
            "FP",
            "FN",
            "tracker_fps",
            "cached_pipeline_fps",
            "unique_predicted_ids",
            "note",
            "metric_path",
        ],
    )
    write_json(OUT_DIR / "qwen_vlm_runs.json", vlm_rows)
    write_csv(
        OUT_DIR / "qwen_vlm_runs.csv",
        vlm_rows,
        [
            "run",
            "status",
            "model_id",
            "image_count",
            "keyframes",
            "max_tokens",
            "answer_chars",
            "oom",
            "track_ref_count",
            "grounded_track_refs",
            "unsupported_track_refs",
            "track_ref_hallucination_rate",
            "error",
            "note",
            "run_dir",
            "answer_json",
        ],
    )
    report_path = write_report(detector_rows, tracking_rows, vlm_rows)
    print(json.dumps({"status": "ok", "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
