# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 EPAM Systems
"""
Custom Script builder module
"""


import yaml
import base64
import os.path
from typing import List
from moulin.yaml_wrapper import YamlValue
from moulin import ninja_syntax


def get_builder(conf: YamlValue, name: str, build_dir: str, src_stamps: List[str],
                generator: ninja_syntax.Writer):
    """
    Return configured CustomScriptBuilder class
    """
    return CustomScriptBuilder(conf, name, build_dir, src_stamps, generator)


def gen_build_rules(generator: ninja_syntax.Writer):
    """
    Generate custom_script build rules for ninja
    """
    cmd = " && ".join([
        "( [ -d $work_dir ] || mkdir $work_dir )",
        "echo '# Code generated by moulin. All manual changes will be lost' > $work_dir/$config_file",
        "echo $b64config | base64 -d - >> $work_dir/$config_file",
    ])
    generator.rule("cs_update_conf",
                   command=cmd,
                   description="Updating config")
    generator.newline()

    cmd = " && ".join([
        "$script $args $work_dir/$config_file"
    ])
    generator.rule("cs_build",
                   command=cmd,
                   description="Running custom script",
                   pool="console",
                   restat=True)
    generator.newline()


class CustomScriptBuilder:
    """
    CustomScriptBuilder class generates Ninja rules to archive files
    """

    def __init__(self, conf: YamlValue, name: str, build_dir: str, src_stamps: List[str],
                 generator: ninja_syntax.Writer):
        self.conf = conf
        self.name = name
        self.generator = generator
        self.src_stamps = src_stamps
        self.build_dir = build_dir

    def gen_build(self):
        """Generate ninja rules launch custom script"""
        work_dir = self.conf.get("work_dir", "script_workdir").as_str
        common_variables = {
            "script": self.conf["script"].as_str,
            "work_dir": work_dir,
        }
        local_conf_file = f"conf-{self.name}.yaml"
        local_conf_target = os.path.join(work_dir, local_conf_file)
        serialized_conf = yaml.serialize(self.conf._node)
        conf_bytes = serialized_conf.encode("utf-8")
        b64bytes = base64.b64encode(conf_bytes)
        b64conf = b64bytes.decode("utf-8")
        targets = self.get_targets()
        additional_deps_node = self.conf.get("additional_deps", None)
        deps = list(self.src_stamps)
        if additional_deps_node:
            deps.extend([d.as_str for d in additional_deps_node])
        deps.append(local_conf_target)
        self.generator.build(local_conf_target, "cs_update_conf", variables=dict(
            common_variables,
            b64config=b64conf,
            config_file=local_conf_file))
        self.generator.newline()

        self.generator.build(f"conf-{self.name}", "phony", local_conf_target)
        self.generator.newline()

        args_node = self.conf.get("args", "")
        if args_node.is_list:
            args = " ".join([x.as_str for x in args_node])
        else:
            args = args_node.as_str
        self.generator.build(targets, "cs_build", deps, variables=dict(
            common_variables,
            config_file=local_conf_file,
            args=args))
        self.generator.newline()

        return targets

    def get_targets(self):
        "Return list of targets that are generated by this build"
        return [t.as_str for t in self.conf["target_images"]]

    def capture_state(self):
        """
        This method should capture state for reproducible builds.
        Luckily, there is nothing to do, as this builder just arhives
        existing files
        """
