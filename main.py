from __future__ import annotations

from tslib.configs.config import config_postprocess, exp_parser
from tslib.exp.exp import run_experiment


def main():
    parser = exp_parser()
    config = config_postprocess(parser.parse_args())
    comp, comp_path = run_experiment(config)
    print(comp)
    print(f"Saved: {comp_path}")


if __name__ == "__main__":
    main()
