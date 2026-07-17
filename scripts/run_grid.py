#!/usr/bin/env python3
"""Run the gradient-inversion reconstruction grid from a YAML config."""
import argparse, logging, sys
sys.path.insert(0, "src")
from flprivacy.config import GridConfig
from flprivacy.pipeline.grid import GridRunner


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/grid.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = GridConfig.from_yaml(args.config)
    out = GridRunner(cfg, device=args.device).run()
    print(f"per-run results written to {out}")


if __name__ == "__main__":
    main()
