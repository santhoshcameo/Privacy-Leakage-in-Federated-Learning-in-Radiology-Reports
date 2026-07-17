"""Grid runner: executes the full tokenizer x batch x dataset x seed grid.

Resumable and deterministic: each (tokenizer, batch, dataset, seed) cell is run
once and appended to a per-run CSV; existing cells are skipped on restart.
"""
from __future__ import annotations
import csv
import logging
from pathlib import Path

from ..config import GridConfig
from ..attack.inversion import GradientInversionAttack
from ..metrics.text_metrics import exact_match, s_bleu, rouge_l

log = logging.getLogger(__name__)
_COLS = ["tokenizer", "batch_size", "dataset", "seed", "n_sequences",
         "exact_accuracy_pct", "sbleu_mean", "rougeL_mean", "elapsed_sec"]


class GridRunner:
    def __init__(self, cfg: GridConfig, device: str = "cuda"):
        self.cfg = cfg
        self.attack = GradientInversionAttack(cfg.attack, device=device)

    def _done(self, out: Path) -> set[tuple]:
        if not out.exists():
            return set()
        with out.open() as f:
            return {(r["tokenizer"], int(r["batch_size"]), r["dataset"], int(r["seed"]))
                    for r in csv.DictReader(f)}

    def run(self) -> Path:
        out = Path(self.cfg.output_dir) / "per_run.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = self._done(out)
        new = not out.exists()
        with out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_COLS)
            if new:
                w.writeheader()
            for tname, tpath in self.cfg.tokenizers.items():
                for bs in self.cfg.batch_sizes:
                    for dname, did in self.cfg.datasets.items():
                        for seed in self.cfg.seeds:
                            if (tname, bs, dname, seed) in done:
                                log.info("skip %s/%s/%s/%s", tname, bs, dname, seed)
                                continue
                            log.info("run  %s bs=%s ds=%s seed=%s", tname, bs, dname, seed)
                            res = self.attack.run(tpath, bs, did, seed)
                            n = len(res.references)
                            e = [exact_match(res.references[i], res.reconstructions[i]) for i in range(n)]
                            sb = [s_bleu(res.references[i], res.reconstructions[i]) for i in range(n)]
                            rl = [rouge_l(res.references[i], res.reconstructions[i]) for i in range(n)]
                            w.writerow({
                                "tokenizer": tname, "batch_size": bs, "dataset": dname,
                                "seed": seed, "n_sequences": n,
                                "exact_accuracy_pct": 100.0 * sum(e) / n,
                                "sbleu_mean": sum(sb) / n, "rougeL_mean": sum(rl) / n,
                                "elapsed_sec": res.elapsed_sec})
                            f.flush()
        return out
