# SPDX-License-Identifier: Apache-2.0
# Copyright 2021 EPAM Systems
"""Module hosting different utility functions"""

import os.path
import sys
import re
from typing import List
from moulin import ninja_syntax

FETCHERDEP_RULE_NAME = "moulin_fetcherdep"


def create_stamp_name(*args):
    """Generate stamp file name based on input keywords"""
    stamp = "-".join(args)
    path = os.path.join(".stamps", stamp.replace("-", "--").replace(os.sep, "-").replace(":", "-"))
    return os.path.abspath(path)


def create_dyndep_fname(component_name: str) -> str:
    return f".moulin_{component_name}.d"


def generate_dyndep_build(generator: ninja_syntax.Writer, component_name: str,
                          actual_targets: List[str]) -> None:
    generator.build(create_dyndep_fname(component_name),
                    FETCHERDEP_RULE_NAME,
                    actual_targets,
                    variables={"name": component_name})
    pass


def escape(val: str) -> str:
    """
    Escape special characters in the input string.

    This function takes an input string `val` and escapes special characters by adding escape
    sequences to them. The following transformations are applied:
    - Double quotes (") are escaped as \\".
    - Dollar signs ($) are escaped as $$.
    Args:
        val (str): The input string to be escaped.
    Returns:
        str: The escaped string.
    """
    result = val
    result = result.replace(r"\"", r"\\\"")
    result = result.replace("$", "$$")
    result = re.sub('(([^\\\\])"|^()")', '\\2\\"', result)
    return result
