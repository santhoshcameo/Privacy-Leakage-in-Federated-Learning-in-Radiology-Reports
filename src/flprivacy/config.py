"""Typed configuration objects, loadable from YAML.

All experiment parameters live here so runs are fully specified and reproducible.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence
import yaml


@dataclass
class AttackConfig:
    """Analytic (imprint) gradient-inversion attack parameters."""
    num_bins: int = 1000              # K: cumulative-bin imprint size
    sequence_length: int = 32         # tokens per reconstructed window
    embedding_dim: int = 96           # m: probe/embedding dimension
    linfunc: str = "randn"            # imprint linear-function init
    ridge_epsilon: float = 1e-8       # numerical-stability term (matches manuscript)
    breaching_attack: str = "imprint"
    breaching_case: str = "10_causal_lang_training"
    breaching_server: str = "malicious-model-rtf"


@dataclass
class DataConfig:
    """Dataset registry entry: a logical name -> a breaching dataset id."""
    name: str
    dataset_id: str


@dataclass
class GridConfig:
    """A full experimental grid: tokenizers x batch sizes x datasets x seeds."""
    tokenizers: dict = field(default_factory=lambda: {
        "gpt2": "gpt2",
        "radbert": "StanfordAIMI/SRR-BERT2BERT-RadBERT",
        "llama2": "meta-llama/Llama-2-7b-hf",
    })
    batch_sizes: Sequence[int] = (64, 128, 256)
    seeds: Sequence[int] = (0, 1, 2, 3, 4)
    datasets: dict = field(default_factory=lambda: {
        "discharge": "dischargesum/discharge_target",
        "diagnosis": "dischargesum/radiology",
        "mimic_cxr": "csv:data/mimic_cxr_validate.csv",
    })
    attack: AttackConfig = field(default_factory=AttackConfig)
    output_dir: str = "results/per_run"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GridConfig":
        raw = yaml.safe_load(Path(path).read_text())
        attack = AttackConfig(**raw.pop("attack", {}))
        return cls(attack=attack, **raw)

    def to_yaml(self, path: str | Path) -> None:
        d = asdict(self)
        Path(path).write_text(yaml.safe_dump(d, sort_keys=False))
