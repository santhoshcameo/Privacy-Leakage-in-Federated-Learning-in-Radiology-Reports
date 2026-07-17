# Privacy Leakage in Federated Learning in Radiology Reports

Reproducible code for the study of **gradient-inversion privacy leakage in
federated learning (FL) for radiology natural-language processing**, with a
controlled ablation across three transformer tokenizers and three batch sizes.

> **Associated manuscript:** *Privacy Leakage in Federated Learning in Radiology
> Reports: A Comparative Evaluation of Tokenizer and Batch-Size Privacy Risks*
> (under review, JMIR Medical Informatics).

---

## Key findings (all reproducible from this repository)

- A malicious FL server can reconstruct **up to ~75% of radiology-report
  sentences** from a single round's gradients under an analytic imprint attack.
- **Batch size is the dominant factor** governing leakage: exact-sentence
  reconstruction falls monotonically from ~65–75% at batch size 64 to ~27–29%
  at batch size 256.
- In a controlled ablation holding the model architecture fixed, **tokenizer
  choice — including a domain-specific clinical tokenizer (RadBERT) — does *not*
  significantly change leakage** (all pairwise comparisons non-significant across
  five seeds; 1 of 27 significant at α = 0.05, consistent with chance).
- **~72–81% of clinical entities are recoverable regardless of tokenizer**
  (MedGemma named-entity-recall analysis), so tokenizer selection cannot be
  relied upon as a privacy safeguard.

The pipeline is **deterministic given the reported seeds**; the released code and
per-seed outputs reproduce every value in the manuscript.

---

## Repository layout

```
.
├── configs/                 # YAML experiment configs (grid, NER)
│   ├── grid.yaml
│   └── ner.yaml
├── src/flprivacy/           # installable package
│   ├── config.py            # typed, YAML-loadable configuration
│   ├── attack/inversion.py  # GradientInversionAttack (imprint / breaching wrapper)
│   ├── metrics/text_metrics.py   # exact accuracy, S-BLEU, ROUGE-L
│   ├── ner/entity_recall.py      # MedGemma clinical-entity recall
│   └── pipeline/grid.py     # resumable grid runner
├── scripts/                 # command-line entry points
│   ├── run_grid.py
│   ├── run_ner.py
│   └── make_figures.py
├── results/per_run/         # released per-seed metric outputs (numbers only)
├── figures/                 # publication figures (regenerated from results)
└── docs/                    # extended reproduction notes
```

---

## Installation

```bash
git clone https://github.com/santhoshcameo/Privacy-Leakage-in-Federated-Learning-in-Radiology-Reports.git
cd Privacy-Leakage-in-Federated-Learning-in-Radiology-Reports
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt

# The analytic imprint attack builds on the `breaching` framework (not on PyPI):
git clone https://github.com/JonasGeiping/breaching.git
pip install -e breaching
```

A CUDA GPU is required to run the attack and the MedGemma NER analysis
(experiments in the manuscript used a single NVIDIA A100-40GB). Figure and table
regeneration from the released `results/per_run/` CSVs needs only CPU.

---

## Reproducing the results

**1 — Reconstruction grid** (exact accuracy, S-BLEU, ROUGE-L):

```bash
python scripts/run_grid.py --config configs/grid.yaml
# -> results/per_run/per_run.csv  (135 rows: 3 tokenizers x 3 batches x 3 datasets x 5 seeds)
```

**2 — Clinical-entity recall** (MedGemma):

```bash
python scripts/run_ner.py --recon results/recon_*.csv --out results/entity_recall.json
```

**3 — Figures and tables**:

```bash
python scripts/make_figures.py   # regenerates Figures 2-4 and appendix violins from results/
```

Every configuration parameter (bins, sequence length, ridge ε, tokenizer paths,
batch sizes, seeds) lives in `configs/*.yaml` — edit there, not in code.

---

## Data availability

- **Dischargesum** (discharge summaries, diagnostic reports): public.
- **MIMIC-CXR** free-text reports: available via **PhysioNet credentialed access**
  under its Data Use Agreement. In accordance with that agreement, MIMIC-CXR text
  and any reconstructed text derived from it are **not redistributed** here; the
  released `results/per_run/` files contain aggregate metrics only (no report
  text). Credentialed users can regenerate the reconstructions with `run_grid.py`.

---

## Threat model

An **active malicious server** that (i) inserts a lightweight analytic imprint
probe (K = 1000 cumulative bins) before the positional embedding, (ii) observes
per-round per-client gradients, and (iii) knows the shared model and tokenizer.
No privacy defenses are enabled — the study quantifies an upper bound on residual
leakage. See the manuscript Methods (Threat Model, Background & Preliminaries).

---

## Citation

```bibtex
@article{parampottupadam_fl_radiology_privacy,
  title   = {Privacy Leakage in Federated Learning in Radiology Reports:
             A Comparative Evaluation of Tokenizer and Batch-Size Privacy Risks},
  author  = {Parampottupadam, Santhosh and others},
  journal = {JMIR Medical Informatics (under review)},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
