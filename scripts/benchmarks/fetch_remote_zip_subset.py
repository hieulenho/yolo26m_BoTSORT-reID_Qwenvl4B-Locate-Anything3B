"""Fetch a contiguous file subset from a large HTTP ZIP using byte ranges."""

from __future__ import annotations

import argparse
import binascii
import bz2
import json
import lzma
import struct
import sys
import zlib
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZIP_BZIP2, ZIP_DEFLATED, ZIP_LZMA, ZIP_STORED, ZipInfo

import requests
from remotezip import RemoteZip
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _session() -> requests.Session:
    retry = Retry(
        total=6,
        backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers["User-Agent"] = "football-tracking-benchmark-fetcher/1.0"
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _decompress(info: ZipInfo, value: bytes) -> bytes:
    if info.compress_type == ZIP_STORED:
        return value
    if info.compress_type == ZIP_DEFLATED:
        return zlib.decompress(value, -15)
    if info.compress_type == ZIP_BZIP2:
        return bz2.decompress(value)
    if info.compress_type == ZIP_LZMA:
        return lzma.decompress(value)
    raise ValueError(f"Unsupported ZIP compression method: {info.compress_type}")


def _safe_relative(name: str, prefix: str) -> Path:
    suffix = name[len(prefix) :].lstrip("/")
    relative = PurePosixPath(suffix or PurePosixPath(name).name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe ZIP member: {name}")
    return Path(*relative.parts)


def fetch_subset(
    *,
    url: str,
    prefix: str,
    output_dir: Path,
    max_files: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    normalized_prefix = prefix.strip("/")
    session = _session()
    with RemoteZip(url, session=session) as archive:
        all_entries = sorted(archive.infolist(), key=lambda row: row.header_offset)
        selected = [
            row
            for row in all_entries
            if not row.is_dir() and row.filename.startswith(normalized_prefix)
        ]
        if max_files is not None:
            selected = selected[:max_files]
        if not selected:
            raise ValueError(f"No files match ZIP prefix: {normalized_prefix}")
        positions = {row.header_offset: index for index, row in enumerate(all_entries)}
        final_index = positions[selected[-1].header_offset]
        next_offset = (
            all_entries[final_index + 1].header_offset
            if final_index + 1 < len(all_entries)
            else archive.start_dir
        )
        start_offset = selected[0].header_offset

    response = session.get(
        url,
        headers={"Range": f"bytes={start_offset}-{next_offset - 1}"},
        timeout=(30, 600),
    )
    response.raise_for_status()
    if response.status_code != 206 or "Content-Range" not in response.headers:
        raise RuntimeError("Remote server did not honor the HTTP byte-range request.")
    block = response.content
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, Any]] = []
    for info in selected:
        offset = info.header_offset - start_offset
        if block[offset : offset + 4] != b"PK\x03\x04":
            raise RuntimeError(f"Invalid local ZIP header for {info.filename}")
        filename_length = struct.unpack_from("<H", block, offset + 26)[0]
        extra_length = struct.unpack_from("<H", block, offset + 28)[0]
        data_start = offset + 30 + filename_length + extra_length
        compressed = block[data_start : data_start + info.compress_size]
        content = _decompress(info, compressed)
        if len(content) != info.file_size:
            raise RuntimeError(f"Size mismatch for {info.filename}")
        if (binascii.crc32(content) & 0xFFFFFFFF) != info.CRC:
            raise RuntimeError(f"CRC mismatch for {info.filename}")
        destination = (output_dir / _safe_relative(info.filename, normalized_prefix)).resolve()
        if output_dir not in destination.parents:
            raise RuntimeError(f"ZIP member escapes output directory: {info.filename}")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        extracted.append(
            {
                "name": info.filename,
                "path": str(destination),
                "size_bytes": info.file_size,
                "crc32": f"{info.CRC:08x}",
            }
        )
    manifest = {
        "schema_version": 1,
        "url": url,
        "prefix": normalized_prefix,
        "http_range": [start_offset, next_offset - 1],
        "downloaded_bytes": len(block),
        "file_count": len(extracted),
        "files": extracted,
    }
    manifest_path = output_dir / "remote_zip_subset.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "ok", "manifest": str(manifest_path), **manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = fetch_subset(
            url=args.url,
            prefix=args.prefix,
            output_dir=args.output_dir,
            max_files=args.max_files,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
