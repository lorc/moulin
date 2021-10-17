#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright 2021 EPAM Systems
"""
Console entry point for rouge (moulin image generator)
"""

import argparse
import os.path
import struct
import shutil
from collections import namedtuple
from typing import Iterator, List, Tuple, Optional
from tempfile import NamedTemporaryFile, TemporaryDirectory
import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode, Node
from moulin.rouge import sfdisk, ext_utils


def traverse_tree(node, fn):
    if isinstance(node, MappingNode):
        for subpair in node.value:
            traverse_tree(subpair[0], fn)
            traverse_tree(subpair[1], fn)
    elif isinstance(node, SequenceNode):
        for subnode in node.value:
            traverse_tree(subnode, fn)
    elif isinstance(node, ScalarNode):
        fn(node)


def get_node(node: MappingNode, name: str) -> Optional[Node]:
    if not isinstance(node, MappingNode):
        raise Exception(f"Not a mapping node {node.start_mark}")
    for subpair in node.value:
        if subpair[0].value == name:
            return subpair[1]
    return None


def get_scalar_node(node: MappingNode, name: str) -> Optional[ScalarNode]:
    value = get_node(node, name)
    if not value:
        return value
    if not isinstance(value, ScalarNode):
        raise Exception(f"Expected scalar value {value.start_mark}")
    return value


def enumarate_mapping(node: MappingNode) -> Iterator[Tuple[ScalarNode, Node]]:
    for subpair in node.value:
        yield (subpair[0], subpair[1])


def update_value(node, newval):
    node.value = newval


_SUFFIXES = {
    "B": 1,
    "KB": 1000,
    "MB": 1000000,
    "GB": 1000000000,
    "TB": 1000000000000,
    "KiB": 1024,
    "MiB": 1024 * 1024,
    "GiB": 1024 * 1024 * 1024,
    "TiB": 1024 * 1024 * 1024 * 1024,
}


def parse_size(node: Node) -> int:
    components = node.value.split(" ")
    if len(components) == 1:
        return int(components[0])
    if len(components) == 2:
        suffix = components[1]
        if not suffix in _SUFFIXES:
            raise Exception(f"Unknown size suffix '{suffix}' {node.start_mark}")
        scaler = _SUFFIXES[suffix]
        return int(components[0]) * scaler
    raise Exception(f"Can't parse size {node.start_mark}")


class BlockEntry():
    "Base class for various block entries"

    def write(self, _file, _offset):
        "write() in base class does nothing"


class GPT(BlockEntry):
    "Represents GUID Partition Table"

    def __init__(self, node: MappingNode):
        entries = []

        partitions = get_node(node, "partitions")
        if not partitions:
            raise Exception(f"Can't find 'partitions' entry {node.start_mark}")
        if not isinstance(partitions, MappingNode):
            raise Exception(f"Excepted mapping 'partitions' {partitions.start_mark}")
        for part_id, part in enumarate_mapping(partitions):
            label: str = part_id.value
            if not isinstance(part, MappingNode):
                raise Exception(f"Excepted mapping node' {part.start_mark}")

            entry_obj, gpt_type = self._process_entry(part)
            entries.append((label, gpt_type, entry_obj))

        self._update_sizes_and_offsets(entries)

    def size(self) -> int:
        "Returns size in bytes"
        return self._size

    @staticmethod
    def _process_entry(node: MappingNode):
        entry_obj = construct_entry(node)
        gpt_type_node = get_node(node, "gpt_type")
        if not gpt_type_node:
            print(f"Warning: no GPT type is provided, using default {node.start_mark}")
            gpt_type = "8DA63339-0007-60C0-C436-083AC8230908"
        else:
            gpt_type = gpt_type_node.value

        return (entry_obj, gpt_type)

    def _update_sizes_and_offsets(self, partitions):
        Partition = namedtuple("Partition", ["start", "size", "label", "block_object"])
        sfdisk_parts: List[sfdisk.Partition] = []
        offset = 0
        for mypart in partitions:
            sfpart = sfdisk.Partition(mypart[0], mypart[1], offset, mypart[2].size())
            offset += mypart[2].size()
            sfdisk_parts.append(sfpart)

        self._sfdisk_parts, self._size = sfdisk.fixup_partition_table(sfdisk_parts)
        self.partitions = [
            Partition(sfdpart.start, sfdpart.size, sfdpart.label, mypart[2])
            for sfdpart, mypart in zip(self._sfdisk_parts, partitions)
        ]

    def write(self, fp, offset):
        if offset == 0:
            sfdisk.write(fp, self._sfdisk_parts)
        else:
            # Write partition into temporary file, then copy it into
            # resulting file
            with NamedTemporaryFile("wb") as tempf:
                tempf.truncate(self._size)
                sfdisk.write(tempf, self._sfdisk_parts)
                ext_utils.dd(tempf, fp, offset)

        for part in self.partitions:
            part.block_object.write(fp, part.start + offset)


class RawImage(BlockEntry):
    "Represents raw image file which needs to be copied as is"

    def __init__(self, node: MappingNode):

        file_node = get_scalar_node(node, "image_path")
        if not file_node:
            raise Exception(f"'image_path' is required {node.start_mark}")
        fname = file_node.value
        if not os.path.exists(fname):
            raise Exception(f"Can't find file '{fname}' {file_node.start_mark}")
        self.fname = fname

        fsize = os.path.getsize(fname)
        size_node = get_scalar_node(node, "size")
        if size_node:
            self._size = parse_size(size_node)
            if fsize > self._size:
                print(f"Warning: file is bigger than partition entry {size_node.start_mark}")
        else:
            self._size = fsize

    def size(self) -> int:
        "Returns size in bytes"
        return self._size

    def write(self, fp, offset):
        ext_utils.dd(self.fname, fp, offset)


class AndroidSparse(BlockEntry):
    "Represents android sparse image file"

    def __init__(self, node: MappingNode):

        file_node = get_scalar_node(node, "image_path")
        if not file_node:
            raise Exception(f"'image_path' is required {node.start_mark}")
        fname = file_node.value
        if not os.path.exists(fname):
            raise Exception(f"Can't find file '{fname}' {file_node.start_mark}")
        self.fname = fname

        fsize = self._read_size(file_node.start_mark)
        size_node = get_scalar_node(node, "size")
        if size_node:
            self._size = parse_size(size_node)
            if fsize > self._size:
                print(f"Warning: file is bigger than partition entry {size_node.start_mark}")
        else:
            self._size = fsize

    def _read_size(self, mark):
        FMT = "<IHHHHIIII"
        MAGIC = 0xed26ff3a
        size = struct.calcsize(FMT)
        with open(self.fname, "rb") as data:
            buf = data.read(size)
            if len(buf) < size:
                raise Exception(
                    f"Not enough data for android sparse header in '{self.fname}' {mark}")
            header = struct.unpack(FMT, buf)
            if header[0] != MAGIC:
                raise Exception(f"Invalid Android header in '{self.fname}' {mark}")
            # blk_sz * total_blks
            return header[5] * header[6]

    def size(self) -> int:
        "Returns size in bytes"
        return self._size

    def write(self, fp, offset):
        with NamedTemporaryFile("w+b", dir=".") as tmpf:
            ext_utils.simg2img(self.fname, tmpf)
            ext_utils.dd(tmpf, fp, offset)


class EmptyEntry(BlockEntry):
    "Represents empty partition"

    def __init__(self, node: MappingNode):

        size_node = get_scalar_node(node, "size")
        if size_node:
            self._size = parse_size(size_node)
        else:
            raise Exception(f"size is mandatory for 'empty' entry {node.start_mark}")

    def size(self) -> int:
        "Returns size in bytes"
        return self._size


class Ext4(BlockEntry):
    "Represents ext4 fs with list of files"

    def __init__(self, node: MappingNode):
        files_node = get_node(node, "files")
        self._files: List[Tuple[str, str]] = []
        if files_node:
            if not isinstance(files_node, MappingNode):
                raise Exception(
                    f"'files' should hold mapping 'remote':'local' {files_node.start_mark}")
            fnode: Tuple[ScalarNode, ScalarNode]
            for fnode in files_node.value:
                print(fnode)
                if not isinstance(fnode, tuple) or not isinstance(
                        fnode[0], ScalarNode) or not isinstance(fnode[1], ScalarNode):
                    raise Exception(f"Expected mapping 'remote':'local' {fnode[0].start_mark}")
                remote_name = fnode[0].value
                local_name = fnode[1].value
                if not os.path.isfile(local_name):
                    raise Exception(f"Can't find file '{local_name}' {fnode[1].start_mark}")
                self._files.append((remote_name, local_name))
        files_size = sum([os.path.getsize(x[1]) for x in self._files]) + 2 * 1024 * 1024
        size_node = get_scalar_node(node, "size")
        if size_node:
            self._size = parse_size(size_node)
            if files_size > self._size:
                raise Exception(" ".join([
                    f"Total files size is {files_size},",
                    f"which bigger than partition size {self._size}", f"{size_node.start_mark}"
                ]))
        else:
            self._size = files_size

    def size(self) -> int:
        "Returns size in bytes"
        return self._size

    def write(self, fp, offset):
        with NamedTemporaryFile() as tempf, TemporaryDirectory() as tempd:
            for remote, local in self._files:
                shutil.copyfile(local, os.path.join(tempd, remote))
            tempf.truncate(self._size)
            ext_utils.mkext4fs(tempf, tempd)
            ext_utils.dd(tempf, fp, offset)


_ENTRY_TYPES = {
    "gpt": GPT,
    "raw_image": RawImage,
    "ext4": Ext4,
    "empty": EmptyEntry,
    "android_sparse": AndroidSparse,
}


def construct_entry(node: MappingNode) -> BlockEntry:
    type_node = get_node(node, "type")
    if not type_node:
        raise Exception(f"Entry 'type' is required {node.start_mark}")

    entry_type: str = type_node.value
    if not entry_type in _ENTRY_TYPES:
        raise Exception(f"Unknown type '{entry_type}' {type_node.start_mark}")

    return _ENTRY_TYPES[entry_type](node)


def main():
    """Console entry point"""

    parser = argparse.ArgumentParser(description='Rouge - image generator')
    parser.add_argument('conf',
                        metavar='build.yaml',
                        type=str,
                        help='YAML file with build description')
    args, extra_opts = parser.parse_known_args()
    conf = yaml.compose(open(args.conf))

    # This is a temporary hack
    traverse_tree(conf, lambda x: update_value(x, x.value.replace("%{MACHINE}", "salvator-x")))
    traverse_tree(conf, lambda x: update_value(x, x.value.replace("%{YOCTOS_WORK_DIR}", "yocto")))
    images_node = get_node(conf, "images")
    if not images_node:
        raise Exception("Can't find 'images' entry in provided config")

    for image_name, image_node in enumarate_mapping(images_node):
        print(image_name)
        main_gpt = GPT(image_node)
        if image_name.value == "full":
            with open("test.img", "wb") as f:
                f.truncate(main_gpt.size())
                main_gpt.write(f, 0)
        print("Total size: ", main_gpt.size())


if __name__ == "__main__":
    main()
