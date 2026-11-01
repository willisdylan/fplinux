# SPDX-License-Identifier: GPL-2.0-only
# ruff: noqa: INP001
"""Prepare Ruio F286 device data from the phone's partition tables."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

from fplinux_cli.device_data.bluetooth_firmware import (
    DOWNLOADED_NV_PARTITION_ID,
    NV_FILES,
    PROTECT_NV_PARTITION_ID,
    REQUIRED_BLUETOOTH_PARTITIONS,
    prepare_bluetooth_from_records,
)
from fplinux_cli.device_data.formats import (
    DeviceDataPreparation,
    PreparedGroup,
    fixed_nv_records,
    redundant_vbm_partitions,
)
from fplinux_cli.device_data.nand_geometry import BLOCK_MAIN_BYTES

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fplinux_cli.device_data.formats import (
        AudioProfileContext,
        NandPartitionReader,
        PhysicalNand,
        RequiredPartition,
    )

_VBM_MAGIC = b"VBM_BOOT"

# Physical extents from the F286 flash partition table. The DSP images are
# shorter than their partitions, so the groups read fixed extents.
GSM_DSP_EXTENT = (0x00260000, 0x00300000)  # fw_modem 0x10000006, whole image
LTE_DSP_EXTENT = (0x00560000, 0x002E4000)  # fw_dsp 0x10000005, stock copy size
# LTE XDSP core A/B images exist only inside the stock AP kernel (partition
# 0x00000003 at 0x01400000, load base 0x80a05e00): (name, VA, 32-bit words),
# with the word counts the stock kernel's RW data at 0x81713f38 gives.
KERNEL_OFFSET, KERNEL_BASE = 0x01400000, 0x80A05E00
XDSP_IMAGES = (
    ("f286-xdspA-boot.bin", 0x815019B4, 0x696),
    ("f286-xdspA-main.bin", 0x814F7F00, 0x26AD),
    ("f286-xdspB-boot.bin", 0x81517684, 0x556),
    ("f286-xdspB-main.bin", 0x8150DDE8, 0x2627),
)


def _vbm_partitions(
    nand: NandPartitionReader, required: Mapping[int, RequiredPartition]
) -> dict[int, tuple[int, int]]:
    """Use the one VBM copy whose table is consistent on its own.

    The two stock VBM copies on the F286 differ in one field: the second gives
    partition 0x4 (mmi_app) a last block of 344, which overlaps the next
    partition at 329. The first copy ends it at 328, matching the 0x920000-byte
    image. Each copy is checked alone with the shared validator, and exactly
    one has to pass.
    """
    copies = [
        block * BLOCK_MAIN_BYTES
        for block in range(nand.block_count)
        if nand.main_bytes(block * BLOCK_MAIN_BYTES, len(_VBM_MAGIC)) == _VBM_MAGIC
    ]
    if len(copies) != 2:
        message = f"expected two VBM partition-table copies, found {len(copies)}"
        raise ValueError(message)
    consistent = []
    for offset in copies:
        try:
            consistent.append(redundant_vbm_partitions(nand, (offset, offset), required))
        except ValueError:
            continue
    if len(consistent) != 1:
        message = f"expected one self-consistent VBM copy, found {len(consistent)}"
        raise ValueError(message)
    return consistent[0]


def _protect_nv_records(partition: bytes, required_sizes: Mapping[int, int]) -> dict[int, bytes]:
    """Read the ProtectNV copy up to the end of its first sorted run.

    On the F286 the stock ProtectNV stream has 224 sorted records, then
    zero padding and records whose IDs go backwards. The Bluetooth records
    before that point are the same as DownloadedNV's, so the stream is
    read up to the first record whose ID does not increase. Every required
    record must still be present at its exact size, and the caller still
    compares them with DownloadedNV.
    """
    cursor, previous_id = 4, 0
    records: dict[int, bytes] = {}
    while cursor + 4 <= len(partition):
        identifier, length = struct.unpack_from("<HH", partition, cursor)
        if identifier <= previous_id or identifier == 0xFFFF:
            break
        padded_length = (length + 3) & ~3
        if cursor + 4 + padded_length > len(partition):
            break
        if identifier in required_sizes:
            records[identifier] = partition[cursor + 4 : cursor + 4 + length]
        previous_id = identifier
        cursor += 4 + padded_length
    for identifier, size in required_sizes.items():
        if len(records.get(identifier, b"")) != size:
            message = f"ProtectNV record {identifier} is missing or has the wrong size"
            raise ValueError(message)
    return records


def _prepare_dsp_firmware(nand: PhysicalNand) -> PreparedGroup:
    """Carry the plaintext GSM/LTE XDSP images as runtime firmware."""
    originals = {
        "f286-gsm-dsp.bin": nand.main_bytes(GSM_DSP_EXTENT[0], GSM_DSP_EXTENT[1]),
        "f286-lte-dsp.bin": nand.main_bytes(LTE_DSP_EXTENT[0], LTE_DSP_EXTENT[1]),
    }
    for name, va, words in XDSP_IMAGES:
        originals[name] = nand.main_bytes(KERNEL_OFFSET + va - KERNEL_BASE, words * 4)
    # The DSP images are executed in place by their cores; no preparation.
    return PreparedGroup(originals=dict(originals), prepared=dict(originals))


def prepare_device_data(
    nand: PhysicalNand, *, audio: AudioProfileContext
) -> DeviceDataPreparation:
    """Prepare the Bluetooth and DSP-firmware groups for this target.

    The F286 has no audio support yet, so the audio context is not used.
    """
    del audio
    extents = _vbm_partitions(nand, REQUIRED_BLUETOOTH_PARTITIONS)
    required_sizes: Mapping[int, int] = {
        identifier: size for identifier, (_filename, size) in NV_FILES.items()
    }
    downloaded = fixed_nv_records(
        nand.partition_bytes(extents[DOWNLOADED_NV_PARTITION_ID]),
        required_sizes,
    )
    protected = _protect_nv_records(
        nand.partition_bytes(extents[PROTECT_NV_PARTITION_ID]),
        required_sizes,
    )
    bluetooth = prepare_bluetooth_from_records(
        nand,
        extents,
        prefix="f286",
        downloaded=downloaded,
        protected=protected,
    )
    return DeviceDataPreparation(
        groups={"bluetooth": bluetooth, "wcn-firmware": _prepare_dsp_firmware(nand)}
    )
