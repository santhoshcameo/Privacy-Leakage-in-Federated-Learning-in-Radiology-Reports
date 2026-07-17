#!/usr/bin/env python3
"""Regenerate all manuscript figures from results/per_run/ and results/ner2_*.json."""
import runpy, pathlib
here = pathlib.Path(__file__).parent
for f in ["make_fig2.py", "make_fig34.py", "make_supp_violins.py"]:
    print(f"running {f} ...")
    runpy.run_path(str(here / f), run_name="__main__")
