#!/usr/bin/env python3
"""Compute MedGemma clinical-entity recall from saved reconstruction CSVs."""
import argparse, csv, json, sys
sys.path.insert(0, "src")
from flprivacy.ner.entity_recall import MedGemmaEntityRecall


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recon", nargs="+", required=True,
                    help="reconstruction CSVs with columns idx,reference,reconstruction")
    ap.add_argument("--model", default="google/medgemma-4b-it")
    ap.add_argument("--out", default="results/entity_recall.json")
    args = ap.parse_args()
    ner = MedGemmaEntityRecall(model_name=args.model)
    report = {}
    for path in args.recon:
        rows = list(csv.DictReader(open(path)))
        texts = [r["reference"] for r in rows] + [r["reconstruction"] for r in rows]
        ents = ner.extract(texts)
        n = len(rows)
        ref = set().union(*ents[:n]) if n else set()
        rec = set().union(*ents[n:]) if n else set()
        report[path] = ner.recall(ref, rec).__dict__
        print(path, report[path])
    json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
