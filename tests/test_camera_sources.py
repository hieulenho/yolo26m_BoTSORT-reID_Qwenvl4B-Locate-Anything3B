from __future__ import annotations

import ipaddress
from typing import Any

import pytest

from football_tracking import camera_sources


def test_bounded_private_network_keeps_small_private_subnet() -> None:
    result = camera_sources.bounded_private_network("192.168.21.4", "255.255.255.0")

    assert result == ipaddress.IPv4Network("192.168.21.0/24")


def test_bounded_private_network_caps_large_network_around_host() -> None:
    result = camera_sources.bounded_private_network("10.20.30.40", "255.0.0.0")

    assert result == ipaddress.IPv4Network("10.20.30.0/24")


@pytest.mark.parametrize(
    ("address", "netmask"),
    [
        ("127.0.0.1", "255.0.0.0"),
        ("169.254.1.2", "255.255.0.0"),
        ("not-an-ip", "255.255.255.0"),
    ],
)
def test_bounded_private_network_rejects_unusable_addresses(
    address: str,
    netmask: str,
) -> None:
    assert camera_sources.bounded_private_network(address, netmask) is None


def test_parse_scan_networks_rejects_large_network() -> None:
    with pytest.raises(ValueError, match="at most 256"):
        camera_sources.parse_scan_networks(["192.168.0.0/16"])


def test_discover_rtsp_cameras_returns_open_hosts_in_address_order(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        camera_sources,
        "_port_is_open",
        lambda address, _port, _timeout: address in {"192.168.50.8", "192.168.50.2"},
    )

    result = camera_sources.discover_rtsp_cameras(
        [ipaddress.IPv4Network("192.168.50.0/28")],
        workers=2,
    )

    assert [item.address for item in result] == ["192.168.50.2", "192.168.50.8"]
    assert all(item.port == 554 for item in result)
