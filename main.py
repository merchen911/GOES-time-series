from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from configs.config import config_postprocess, exp_parser
from exp.exp import run_experiment


def main():
    parser = exp_parser()
    config = config_postprocess(parser.parse_args())
    comp, comp_path = run_experiment(config)
    print(comp)
    print(f"Saved: {comp_path}")


if __name__ == "__main__":
    main()
