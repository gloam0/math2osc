from typing import Dict, Mapping, Optional

import numpy as np

from .language import DEFAULT_FUNC_REGISTRY, FunctionSpec


def build_eval_env(
    func_registry: Mapping[str, FunctionSpec] = DEFAULT_FUNC_REGISTRY,
    *,
    sr: Optional[float] = None,
    dt: Optional[float] = None,
    extra: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    env = {spec.target: spec.impl for spec in func_registry.values()}
    env["np"] = np
    env["pi"] = np.pi
    env["tau"] = 2 * np.pi
    env["e"] = np.e
    if sr is not None:
        env["sr"] = float(sr)
    if dt is None and sr is not None:
        dt = 1.0 / float(sr)
    if dt is not None:
        env["dt"] = float(dt)
    if extra:
        env.update(extra)
    return env
