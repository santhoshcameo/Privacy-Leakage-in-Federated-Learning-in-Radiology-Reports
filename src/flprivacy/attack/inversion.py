"""Gradient-inversion attack on a federated transformer.

Thin, typed wrapper around the `breaching` analytic imprint attack, configured
to the malicious-server threat model described in the manuscript: a lightweight
linear "imprint" probe (K cumulative bins) is inserted before the positional
embedding, and per-round gradients are analytically inverted to recover
token embeddings, which are then decoded against the active tokenizer.

The wrapper isolates all breaching-specific setup so the rest of the codebase is
framework-agnostic and the exact attack configuration is reproducible from the
`AttackConfig`.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
import torch

from ..config import AttackConfig


@dataclass
class ReconstructionResult:
    """Reference and reconstructed windows plus per-window elapsed time."""
    references: list[str]
    reconstructions: list[str]
    elapsed_sec: float


class GradientInversionAttack:
    """Analytic imprint gradient-inversion attack for one (tokenizer, batch, data) cell."""

    def __init__(self, cfg: AttackConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = device

    def _build_case(self, tokenizer_path: str, batch_size: int, dataset_id: str):
        import breaching  # imported lazily so the package installs without a GPU
        overrides = [
            f"attack={self.cfg.breaching_attack}",
            f"case={self.cfg.breaching_case}",
            f"case/server={self.cfg.breaching_server}",
        ]
        os.environ["JMIR_DATASET_ID"] = dataset_id
        os.environ["JMIR_N_SAMPLES"] = str(batch_size)
        conf = breaching.get_config(overrides=overrides)
        conf.case.user.num_data_points = batch_size
        conf.case.user.user_idx = 0
        conf.case.data.shape = [self.cfg.sequence_length]
        conf.case.data.default_clients = 1
        if hasattr(conf.case.server, "model_modification"):
            conf.case.server.model_modification.num_bins = self.cfg.num_bins
            conf.case.server.model_modification.position = None
            conf.case.server.model_modification.linfunc = self.cfg.linfunc
        if hasattr(conf.case.server, "has_external_data"):
            conf.case.server.has_external_data = False
        conf.case.data.name = "radiology"
        conf.case.data.tokenizer = tokenizer_path
        conf.case.data.vocab_size = None
        conf.case.data.batch_size = batch_size
        return conf

    def run(self, tokenizer_path: str, batch_size: int, dataset_id: str,
            seed: int) -> ReconstructionResult:
        """Execute a single-round attack and return decoded reference/reconstruction."""
        import time
        import breaching
        from breaching.analysis.analysis import compute_text_order

        torch.manual_seed(seed)
        conf = self._build_case(tokenizer_path, batch_size, dataset_id)
        setup = dict(device=torch.device(self.device),
                     dtype=getattr(torch, conf.case.impl.dtype))
        t0 = time.time()
        user, server, model, loss_fn = breaching.cases.construct_case(conf.case, setup)
        attacker = breaching.attacks.prepare_attack(server.model, server.loss,
                                                    conf.attack, setup)
        payload = server.distribute_payload()
        shared_data, true_user_data = user.compute_local_updates(payload)
        recon, _ = attacker.reconstruct([payload], [shared_data],
                                        server.secrets, dryrun=conf.dryrun)
        order = compute_text_order(recon, true_user_data,
                                   vocab_size=conf.case.data.vocab_size)
        recon["data"] = recon["data"][order]
        elapsed = time.time() - t0

        tok = user.dataloader.dataset.tokenizer
        refs = tok.batch_decode(true_user_data["data"].cpu(),
                                clean_up_tokenization_spaces=True)
        recs = tok.batch_decode(recon["data"].cpu(),
                                clean_up_tokenization_spaces=True)
        n = min(len(refs), len(recs))
        return ReconstructionResult(references=list(refs[:n]),
                                    reconstructions=list(recs[:n]),
                                    elapsed_sec=elapsed)
