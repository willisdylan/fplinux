# SPDX-License-Identifier: GPL-2.0-only
"""Let one UMS9117 board replace the platform boot layout without moving the others."""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest
from fplinux_cli import common
from fplinux_cli.build.bootstrap.recipe import effective_boot_layout
from fplinux_cli.manifests import targets
from fplinux_cli.manifests.platforms import load_platform

LAYOUT_48MB = {
    "ram_base": 0x80000000,
    "ram_size": 0x03000000,
    "kernel_load": 0x81000000,
    "kernel_entry": 0x81000000,
    "kernel_size": 0x01200000,
    "fdt_load": 0x82E00000,
    "fdt_size": 0x00010000,
    "framebuffer": 0x82F00000,
    "framebuffer_size": 0x00100000,
    "timer_hz": 1000,
}


def _board(layout: dict[str, int], linux_size: int = 0x02C00000) -> dict[str, Any]:
    raw = common.load_toml(common.ROOT / "targets/nokia-ta1618/target.toml")
    raw["bootstrap"]["layout"] = layout
    raw["linux"]["memory"] = {"base": 0x80200000, "size": linux_size}
    return raw


def test_board_layout_replaces_the_platform_default() -> None:
    """The bootstrap is built for the board's own layout."""
    with mock.patch.object(targets, "load_toml", return_value=_board(LAYOUT_48MB)):
        config = targets.load_target("nokia-ta1618")
    assert config["bootstrap"]["layout"] == LAYOUT_48MB
    assert effective_boot_layout(config) == LAYOUT_48MB


def test_other_boards_keep_the_platform_default() -> None:
    """A board without its own layout still uses the shared one."""
    config = targets.load_target("nokia-ta1618")
    assert config["bootstrap"]["layout"] == load_platform("ums9117")["bootstrap"]["layout"]


def test_board_layout_must_hold_the_linux_memory() -> None:
    """A layout smaller than the board's Linux range is refused."""
    with (
        mock.patch.object(
            targets, "load_toml", return_value=_board(LAYOUT_48MB, linux_size=0x03C00000)
        ),
        pytest.raises(SystemExit, match="must fit inside physical RAM"),
    ):
        targets.load_target("nokia-ta1618")


def test_board_layout_is_validated_like_the_platform_one() -> None:
    """A partial table cannot inherit the missing fields from the platform."""
    partial = {key: value for key, value in LAYOUT_48MB.items() if key != "timer_hz"}
    with (
        mock.patch.object(targets, "load_toml", return_value=_board(partial)),
        pytest.raises(SystemExit, match="target bootstrap layout"),
    ):
        targets.load_target("nokia-ta1618")


def test_ruio_f286_boots_inside_its_48mb() -> None:
    """The Ruio F286 declares the 48 MB layout its bootstrap needs."""
    assert targets.load_target("ruio-f286")["bootstrap"]["layout"] == LAYOUT_48MB
