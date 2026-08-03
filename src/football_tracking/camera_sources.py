"""Discover readable local webcams and RTSP endpoints on local IPv4 networks."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import cv2

from football_tracking.video_capture import open_video_capture


@dataclass(frozen=True)
class LocalCameraSource:
    kind: str
    device_index: int
    backend: str
    width: int
    height: int
    reported_fps: float

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RtspCameraSource:
    kind: str
    address: str
    port: int

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


def probe_local_camera(
    index: int,
    *,
    read_attempts: int = 3,
) -> LocalCameraSource | None:
    """Return metadata only when a local camera can deliver a non-empty frame."""

    capture, backend_name = open_video_capture(index)
    if capture is None or backend_name is None:
        return None
    try:
        frame = None
        for _ in range(max(1, read_attempts)):
            ok, candidate = capture.read()
            if ok and candidate is not None and candidate.size:
                frame = candidate
                break
        if frame is None:
            return None
        height, width = frame.shape[:2]
        return LocalCameraSource(
            kind="local",
            device_index=index,
            backend=backend_name,
            width=int(width),
            height=int(height),
            reported_fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        )
    finally:
        capture.release()


def discover_local_cameras(max_index: int = 5) -> tuple[LocalCameraSource, ...]:
    """Enumerate readable DirectShow/default camera indices in stable index order."""

    cameras = (
        probe_local_camera(index)
        for index in range(max(0, int(max_index)) + 1)
    )
    return tuple(camera for camera in cameras if camera is not None)


def bounded_private_network(
    address: str,
    netmask: str,
    *,
    widest_prefix: int = 24,
) -> ipaddress.IPv4Network | None:
    """Return a private scan network capped to a small subnet around the host."""

    try:
        host = ipaddress.IPv4Address(address)
        network = ipaddress.IPv4Network((address, netmask), strict=False)
    except (ipaddress.AddressValueError, ipaddress.NetmaskValueError):
        return None
    if host.is_loopback or host.is_link_local or host.is_multicast or not host.is_private:
        return None
    if network.prefixlen < widest_prefix:
        network = ipaddress.IPv4Network(f"{host}/{widest_prefix}", strict=False)
    if network.prefixlen > 30:
        return None
    return network


def local_private_networks() -> tuple[ipaddress.IPv4Network, ...]:
    """Read active adapter addresses and return bounded private IPv4 networks."""

    try:
        import psutil
    except ImportError as exc:  # pragma: no cover - guarded by runtime requirements
        raise RuntimeError(
            "Camera discovery requires psutil. Run .\\scripts\\setup_webcam.ps1."
        ) from exc

    interface_stats = psutil.net_if_stats()
    networks: set[ipaddress.IPv4Network] = set()
    for interface, addresses in psutil.net_if_addrs().items():
        stats = interface_stats.get(interface)
        if stats is not None and not stats.isup:
            continue
        for item in addresses:
            if item.family != socket.AF_INET or not item.netmask:
                continue
            network = bounded_private_network(item.address, item.netmask)
            if network is not None:
                networks.add(network)
    return tuple(sorted(networks, key=lambda value: (int(value.network_address), value.prefixlen)))


def _port_is_open(address: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((address, port), timeout=timeout):
            return True
    except OSError:
        return False


def discover_rtsp_cameras(
    networks: Iterable[ipaddress.IPv4Network] | None = None,
    *,
    port: int = 554,
    timeout: float = 0.2,
    workers: int = 64,
) -> tuple[RtspCameraSource, ...]:
    """Find hosts accepting RTSP TCP connections on bounded local networks."""

    selected_networks = tuple(local_private_networks() if networks is None else networks)
    addresses = sorted(
        {str(host) for network in selected_networks for host in network.hosts()},
        key=lambda value: int(ipaddress.IPv4Address(value)),
    )
    if not addresses:
        return ()
    worker_count = max(1, min(int(workers), len(addresses), 128))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = executor.map(
            lambda address: (address, _port_is_open(address, int(port), float(timeout))),
            addresses,
        )
    return tuple(
        RtspCameraSource(kind="rtsp", address=address, port=int(port))
        for address, is_open in results
        if is_open
    )


def parse_scan_networks(values: Iterable[str]) -> tuple[ipaddress.IPv4Network, ...]:
    """Parse explicit CIDR networks for deterministic discovery and diagnostics."""

    networks: set[ipaddress.IPv4Network] = set()
    for value in values:
        parsed = ipaddress.ip_network(str(value).strip(), strict=False)
        if not isinstance(parsed, ipaddress.IPv4Network):
            raise ValueError(f"Only IPv4 scan networks are supported: {value}")
        if parsed.num_addresses > 256:
            raise ValueError(f"Scan network must contain at most 256 addresses: {value}")
        networks.add(parsed)
    return tuple(sorted(networks, key=lambda item: int(item.network_address)))
