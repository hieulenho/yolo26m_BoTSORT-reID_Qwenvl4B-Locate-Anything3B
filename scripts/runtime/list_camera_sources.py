"""List local webcams and discover RTSP cameras as one logical source inventory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

from football_tracking.camera_sources import (
    discover_local_cameras,
    discover_rtsp_cameras,
    parse_scan_networks,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-local-index", type=int, default=5)
    parser.add_argument("--skip-local", action="store_true")
    parser.add_argument("--skip-network", action="store_true")
    parser.add_argument("--network", action="append", default=[])
    parser.add_argument("--rtsp-port", type=int, default=554)
    parser.add_argument("--timeout", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    if args.max_local_index < 0:
        parser.error("--max-local-index must be non-negative")
    if not 1 <= args.rtsp_port <= 65535:
        parser.error("--rtsp-port must be between 1 and 65535")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    try:
        local = () if args.skip_local else discover_local_cameras(args.max_local_index)
        if args.skip_network:
            rtsp = ()
        else:
            networks = parse_scan_networks(args.network) if args.network else None
            rtsp = discover_rtsp_cameras(
                networks,
                port=args.rtsp_port,
                timeout=args.timeout,
                workers=args.workers,
            )
    except (RuntimeError, ValueError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2

    records: list[dict[str, Any]] = []
    for source in (*local, *rtsp):
        record = source.as_record()
        record["camera_index"] = len(records)
        records.append(record)
    print(
        json.dumps(
            {
                "status": "ok",
                "camera_count": len(records),
                "sources": records,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
