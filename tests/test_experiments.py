from pathlib import Path

import yaml

from gwmaxcut.experiments import REGISTRY

QUICK = yaml.safe_load((Path(__file__).parents[1] / "configs" / "quick.yaml").read_text())


def test_every_experiment_runs_on_the_quick_config():
    for name, fn in REGISTRY.items():
        out = fn(QUICK[name], QUICK["seed"], QUICK["solver"], log=lambda *_: None)
        df = out[0] if isinstance(out, tuple) else out
        assert len(df) > 0, name
