"""Load the upstream PolyFly flatness reference implementation, without importing PolyFly.

Why this exists rather than a plain ``import``: ``poly_fly.optimal_planner.planner`` imports CasADi,
Torch, PIL and Matplotlib at module scope, and CasADi is not installed in this project's
environment (it lives in the separate planning container — see F1 §3). Importing the module is
therefore not possible here.

Rather than hand-copying the upstream math into the test — which would make the cross-check
circular, since a transcription error would be invisible — we parse the upstream *source files*
with :mod:`ast`, lift out the exact function definitions we need, and execute those definitions in
a namespace containing only NumPy and SciPy. The bytes that run are literally upstream's, so the
comparison is against upstream's code and not against our reading of it. Nothing under the upstream
checkout is modified or written to.

The functions lifted are:

* ``Planner.differential_flatness``            (``optimal_planner/planner.py``)
* ``rpy_from_a_to_b``, ``rotation_matrix_from_a_to_b``, ``_skew``  (``utils/utils.py``)

If upstream ever changes their signatures or names, this module raises rather than silently
comparing against something else.
"""

from __future__ import annotations

import ast
import copy
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

DEFAULT_POLYFLY_DIR = Path("/home/praneetho/Desktop/polyfly_ral")

_PLANNER_REL = Path("src/poly_fly/optimal_planner/planner.py")
_UTILS_REL = Path("src/poly_fly/utils/utils.py")


class UpstreamUnavailable(RuntimeError):
    """Raised when the upstream checkout is absent or does not contain the expected symbols."""


@dataclass(frozen=True)
class UpstreamParams:
    """Duck-typed stand-in for upstream's ``params`` object.

    ``differential_flatness`` only touches these three attributes.
    """

    mass_quad: float = 0.715
    mass_load: float = 0.163
    cable_length: float = 0.567


def polyfly_dir() -> Path:
    """Locate the upstream checkout, honoring ``POLYFLY_DIR`` as upstream itself does."""
    return Path(os.environ.get("POLYFLY_DIR", DEFAULT_POLYFLY_DIR))


def _extract(source: str, name: str, class_name: str | None = None) -> ast.FunctionDef:
    tree = ast.parse(source)
    body = tree.body
    if class_name is not None:
        matches = [n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name]
        if not matches:
            raise UpstreamUnavailable(f"upstream class {class_name!r} not found")
        body = matches[0].body
    for node in body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            stripped = copy.deepcopy(node)
            stripped.decorator_list = []  # drop @staticmethod so it becomes a plain function
            return ast.fix_missing_locations(stripped)
    where = f"{class_name}." if class_name else ""
    raise UpstreamUnavailable(f"upstream function {where}{name!r} not found")


@lru_cache(maxsize=1)
def load_upstream_differential_flatness() -> Callable:
    """Return upstream's ``differential_flatness(x, v, a, jrk, params)`` as a callable."""
    import numpy as np
    from scipy.spatial.transform import Rotation as Rot

    root = polyfly_dir()
    planner_path = root / _PLANNER_REL
    utils_path = root / _UTILS_REL
    for path in (planner_path, utils_path):
        if not path.is_file():
            raise UpstreamUnavailable(f"upstream source not found: {path}")

    planner_src = planner_path.read_text()
    utils_src = utils_path.read_text()

    module = ast.Module(
        body=[
            _extract(utils_src, "_skew"),
            _extract(utils_src, "rotation_matrix_from_a_to_b"),
            _extract(utils_src, "rpy_from_a_to_b"),
            _extract(planner_src, "differential_flatness", class_name="Planner"),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)

    namespace: dict[str, object] = {"np": np, "Rot": Rot, "__name__": "polyfly_upstream_ref"}
    exec(compile(module, filename=str(planner_path), mode="exec"), namespace)  # noqa: S102

    fn = namespace["differential_flatness"]
    if not callable(fn):
        raise UpstreamUnavailable("extracted differential_flatness is not callable")
    return fn
