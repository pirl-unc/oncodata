# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""cancerdata is the base of the dependency pyramid.

It is the upstream source of truth for cancer reference data (expression, HPA
protein/RNA, the CTA definition, the cancer-type ontology, anti-PD-1 ORR, TMB) and
must never import its consumers — data and logic flow only downward. This guard
fails the build if any consumer leaks into the shipped package.
"""

import ast
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[1] / "cancerdata"

# Downstream consumers that depend on cancerdata; importing any of them would
# invert the dependency pyramid.
_CONSUMERS = {"pirlygenes", "tsarina", "hitlist", "trufflepig"}


def _imported_top_level_modules(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        # node.level == 0 skips relative (intra-package) imports.
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


def test_package_imports_no_consumer():
    offenders = []
    for py in sorted(_PACKAGE.rglob("*.py")):
        for mod in _imported_top_level_modules(py):
            if mod in _CONSUMERS:
                offenders.append(f"{py.relative_to(_PACKAGE)} imports {mod}")
    assert not offenders, "cancerdata must not import its consumers:\n  " + "\n  ".join(offenders)
