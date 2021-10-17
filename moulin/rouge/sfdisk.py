# SPDX-License-Identifier: Apache-2.0
# Copyright 2021 EPAM Systems
"""
sfdisk interface/wrapper for rouge image builder
"""

from collections import namedtuple
from typing import List, Tuple
from math import ceil
from operator import methodcaller
import subprocess


def _div_up(num: int, dem: int) -> int:
    return int(ceil(num / dem))


def _sect(val, sector_size):
    return _div_up(val, sector_size)


def _align(val, align):
    return _div_up(val, align) * align


DEFAULT_ALIGNMENT = 1 * 1024 * 1024  # 1 MiB


class Partition(namedtuple("Partition", ["label", "guid_type", "start", "size"])):
    "Represents partition entry"

    def to_script(self, sector_size=512):
        "Convert self to sfdisk script line"

        return ", ".join([
            f"start={_sect(self.start, sector_size)}", f"size={_sect(self.size, sector_size)}",
            f"type={self.guid_type}", f"name={self.label}"
        ])


def _check_sfdisk():
    # We are checking result explicitely
    # pylint: disable=subprocess-run-check
    ret = subprocess.run(["which", "sfdisk"], stdout=subprocess.DEVNULL)
    if ret.returncode != 0:
        raise Exception("Please make sure that 'sfdisk' is installed")


def _sfdisk_header():
    return "\n".join(["label: gpt", "unit: sectors"])


def fixup_partition_table(partitions: List[Partition],
                          sector_size=512) -> Tuple[List[Partition], int]:
    """
    Return fixed partition table so it can be really written to disk.
    Also return total size of partition.
    """
    start_offset = 0
    if _sect(partitions[0].start, sector_size) < 2048:
        start_offset = 2048
    end = start_offset
    ret = []
    for part in partitions:
        start = (_sect(part.start, sector_size) + start_offset) * sector_size
        start = _align(start, DEFAULT_ALIGNMENT)  # Align to 1 MB
        size = _sect(part.size, sector_size) * sector_size
        ret.append(part._replace(start=start, size=size))
        end = part.start + part.size

    # Account for GPT copy
    return ret, end + 16 * 1024 * 1024


def write(fileo, partitions: List[Partition]):
    "Write partitions to a file"
    _check_sfdisk()

    # Generate sfdisk script file
    script = _sfdisk_header() + "\n\n"
    script += "\n".join(map(methodcaller("to_script"), partitions))

    print(f"Creating GPT partition in {fileo.name}")
    subprocess.run(["sfdisk", fileo.name],
                   input=bytes(script, 'UTF-8'),
                   check=True,
                   stdout=subprocess.DEVNULL)
