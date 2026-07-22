"""Download a small, licensed multi-domain video set from Wikimedia Commons."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import yaml

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "football-tracking-research/1.0 (multidomain benchmark downloader)"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/multidomain_public_samples.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _open_with_backoff(request: urllib.request.Request, *, timeout: int):
    for attempt in range(5):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = (
                float(retry_after) if retry_after and retry_after.isdigit() else 5.0 * (2**attempt)
            )
            time.sleep(min(delay, 60.0))
    raise RuntimeError("Unreachable download retry state.")


def _request_json(parameters: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode(parameters)
    request = urllib.request.Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    with _open_with_backoff(request, timeout=60) as response:
        return json.load(response)


def _video_info(file_title: str) -> dict[str, Any]:
    payload = _request_json(
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "videoinfo",
            "titles": f"File:{file_title}",
            "viprop": "url|size|mime|derivatives|extmetadata",
        }
    )
    pages = payload.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        raise RuntimeError(f"Wikimedia file does not exist: {file_title}")
    rows = pages[0].get("videoinfo", [])
    if not rows:
        raise RuntimeError(f"Wikimedia returned no video metadata: {file_title}")
    return dict(rows[0])


def _select_video_url(info: dict[str, Any], target_width: int) -> tuple[str, dict[str, Any]]:
    derivatives = [
        dict(row)
        for row in info.get("derivatives", [])
        if str(row.get("type", "")).startswith("video/") and row.get("src")
    ]
    candidates = [row for row in derivatives if int(row.get("width", 0)) <= target_width]
    selected = max(candidates, key=lambda row: int(row.get("width", 0))) if candidates else None
    if selected is None and derivatives:
        selected = min(derivatives, key=lambda row: int(row.get("width", 0)) or 10**9)
    if selected is not None:
        return str(selected["src"]), selected
    return str(info["url"]), {
        "width": info.get("width"),
        "height": info.get("height"),
        "type": info.get("mime"),
        "transcodekey": "original",
    }


def _download(url: str, destination: Path) -> tuple[str, int]:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    size = 0
    try:
        with _open_with_backoff(request, timeout=120) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return digest.hexdigest(), size


def _metadata_value(metadata: dict[str, Any], name: str) -> str:
    value = metadata.get(name, {})
    if isinstance(value, dict):
        value = value.get("value", "")
    plain = re.sub(r"<[^>]+>", " ", html.unescape(str(value)))
    if "Ã" in plain or "â" in plain:
        try:
            plain = plain.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return " ".join(plain.split())


def _probe_video(path: Path) -> dict[str, Any]:
    import cv2  # type: ignore[import-not-found]

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Downloaded video cannot be decoded: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    duration = frame_count / fps if fps > 0.0 else 0.0
    if fps <= 0.0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Downloaded video has invalid stream metadata: {path}")
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "width": width,
        "height": height,
    }


def download_samples(config_path: Path, output_dir: Path | None, overwrite: bool) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = (output_dir or Path(config["download_root"])).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target_width = int(config.get("target_width", 960))
    minimum_duration = float(config.get("minimum_duration_seconds", 30.0))
    records: list[dict[str, Any]] = []
    for sample in config.get("samples", []):
        destination = root / str(sample["output_name"])
        if destination.exists() and not overwrite:
            sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
            size = destination.stat().st_size
            info = _video_info(str(sample["file_title"]))
            selected_url, derivative = _select_video_url(info, target_width)
        else:
            info = _video_info(str(sample["file_title"]))
            selected_url, derivative = _select_video_url(info, target_width)
            sha256, size = _download(selected_url, destination)
        video = _probe_video(destination)
        if video["duration_seconds"] < minimum_duration:
            raise RuntimeError(
                f"Sample {sample['sample_id']} is only "
                f"{video['duration_seconds']:.2f}s; minimum is {minimum_duration:.2f}s."
            )
        ext = dict(info.get("extmetadata", {}))
        records.append(
            {
                "sample_id": sample["sample_id"],
                "path": str(destination),
                "sha256": sha256,
                "bytes": size,
                "source_page": sample["source_page"],
                "download_url": selected_url,
                "derivative": derivative,
                "video": video,
                "selection_reason": sample.get("selection_reason"),
                "license": _metadata_value(ext, "LicenseShortName"),
                "license_url": _metadata_value(ext, "LicenseUrl"),
                "artist": _metadata_value(ext, "Artist"),
                "credit": _metadata_value(ext, "Credit"),
                "ground_truth": sample.get("ground_truth", {}),
            }
        )
        time.sleep(2.0)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "config": str(config_path.resolve()),
        "sample_count": len(records),
        "samples": records,
    }
    manifest_path = root / "samples_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": "ok", "manifest": str(manifest_path), "samples": records}


def main() -> int:
    args = _arguments()
    result = download_samples(args.config, args.output_dir, args.overwrite)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
