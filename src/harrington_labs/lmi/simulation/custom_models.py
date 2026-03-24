"""Custom physics model plugin system.

Users can drop Python files into ``data/custom_models/`` to extend the
Interaction Analyzer with custom absorption, nonlinear, or damage models.

Each plugin must define:

    MODEL_NAME: str          — display name
    MODEL_DESCRIPTION: str   — one-liner shown in the UI
    MODEL_VERSION: str       — semver string (optional, defaults to "0.1.0")

    def compute(
        laser: dict,
        material: dict,
        thickness_m: float,
        z_position_m: float,
    ) -> dict:
        '''Return a dict of computed metrics.

        Keys become metric labels in the UI.  Values can be:
        - float / int  → displayed as a metric card
        - dict with {"x": [...], "y": [...], "label": str}  → plotted
        - str  → displayed as info text
        '''
        ...

``laser`` and ``material`` are plain dicts produced by
``dataclasses.asdict()`` so models have zero dependency on the
harrington-labs LMI subpackage.

See ``data/custom_models/_example_fresnel.py`` for a working reference.
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

MODELS_DIR = Path("data/custom_models")


@dataclass
class CustomModel:
    """A loaded custom physics model."""

    name: str
    description: str
    version: str
    compute: Callable[..., dict[str, Any]]
    filepath: Path
    error: str = ""


def load_models(models_dir: Path | None = None) -> list[CustomModel]:
    """Discover and load all .py models in the custom models directory.

    Silently skips files starting with ``_`` (treated as examples / helpers)
    unless they define ``MODEL_NAME``.  Returns a list of loaded models,
    including any that failed to load (with ``error`` populated).
    """
    d = models_dir or MODELS_DIR
    if not d.exists():
        return []

    models: list[CustomModel] = []

    for path in sorted(d.glob("*.py")):
        if path.name.startswith("__"):
            continue

        try:
            spec = importlib.util.spec_from_file_location(
                f"custom_model_{path.stem}", str(path),
            )
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            # Don't pollute sys.modules
            spec.loader.exec_module(module)

            name = getattr(module, "MODEL_NAME", None)
            if name is None:
                # Skip files that don't declare MODEL_NAME
                continue

            compute_fn = getattr(module, "compute", None)
            if compute_fn is None:
                models.append(CustomModel(
                    name=name,
                    description=getattr(module, "MODEL_DESCRIPTION", ""),
                    version=getattr(module, "MODEL_VERSION", "0.1.0"),
                    compute=lambda **kw: {},
                    filepath=path,
                    error="Missing compute() function",
                ))
                continue

            models.append(CustomModel(
                name=name,
                description=getattr(module, "MODEL_DESCRIPTION", ""),
                version=getattr(module, "MODEL_VERSION", "0.1.0"),
                compute=compute_fn,
                filepath=path,
            ))

        except Exception:
            models.append(CustomModel(
                name=path.stem,
                description="Failed to load",
                version="0.0.0",
                compute=lambda **kw: {},
                filepath=path,
                error=traceback.format_exc(),
            ))

    return models


def run_model(
    model: CustomModel,
    laser_dict: dict,
    material_dict: dict,
    thickness_m: float,
    z_position_m: float = 0.0,
) -> dict[str, Any]:
    """Execute a custom model safely, catching exceptions."""
    try:
        return model.compute(
            laser=laser_dict,
            material=material_dict,
            thickness_m=thickness_m,
            z_position_m=z_position_m,
        )
    except Exception:
        return {"error": traceback.format_exc()}
