from __future__ import annotations

from tslib.configs.config import config_postprocess, exp_parser
from tslib.exp.exp import comparison_paths, run_experiment, save_dir_name


def main():
    parser = exp_parser()
    config = config_postprocess(parser.parse_args())
    regression, event = run_experiment(config)
    reg_path, event_path = comparison_paths(save_dir_name(config))
    print(f"Saved: {reg_path} ({len(regression)} rows x {regression.shape[1]} columns)")
    print(f"Saved: {event_path} ({len(event)} rows x {event.shape[1]} columns)")


if __name__ == "__main__":
    main()
