"""Interactive pre-test: probe each benchmark cell's models, project runtime,
double-check models over a threshold, and write an approved manifest for the
driver (``tslib.benchmark.driver --manifest``).

Projection basis: ``per_epoch_time * nominal_epochs`` (actual training uses
max_epochs + EarlyStopping, so this is a yardstick, not the real duration).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from tslib.benchmark import driver
from tslib.configs.config import exp_parser, config_postprocess
from tslib.data.loader import DataModule
from tslib.model import build_model
from tslib.exp.losses import build_loss


def estimate_train_hours(sec_per_batch: float, n_batches: float,
                         epochs: int) -> float:
    return float(sec_per_batch) * float(n_batches) * float(epochs) / 3600.0


def decide(projected_hours: float, threshold_hours: float, prompt_fn) -> bool:
    """Auto-approve at/under threshold; otherwise defer to prompt_fn(hours)."""
    if projected_hours <= threshold_hours:
        return True
    return bool(prompt_fn(projected_hours))


def _default_prompt(cell, model):
    def ask(hours):
        ans = input(f"[{cell['track']} seq{cell['seq_len']} "
                    f"pred{cell['pred_len']} f{cell['fold']} {cell['strategy']}] "
                    f"{model} ~{hours:.1f}h projected — proceed? [y/N] ")
        return ans.strip().lower() in ("y", "yes")
    return ask


def probe_sec_per_batch(model, train_loader, config, probe_batches: int) -> float:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    model.train()
    crit = build_loss(config)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr,
                           weight_decay=config.weight_decay)
    durations = []
    it = iter(train_loader)
    for i in range(probe_batches + 1):  # +1 warmup, dropped
        try:
            x, y = next(it)
        except StopIteration:
            break
        x, y = x.to(dev), y.to(dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = model(x)
        loss = crit(pred, y[:, :pred.shape[1], :])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if dev == "cuda":
            torch.cuda.synchronize()
        if i > 0:
            durations.append(time.perf_counter() - t0)
    return statistics.median(durations) if durations else float("inf")


def write_manifest(approved, params, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"params": params, "approved": approved}, f, indent=2)


def _config_for(cell, model, nominal_epochs):
    argv = driver.cell_argv(cell, epochs=nominal_epochs, models=[model])
    return config_postprocess(exp_parser().parse_args(argv))


def main(argv=None):
    p = argparse.ArgumentParser(description="benchmark runtime pre-test")
    p.add_argument("--tracks", nargs="+", default=list(driver.TRACKS),
                   choices=list(driver.TRACKS))
    p.add_argument("--seq-lens", nargs="+", type=int, default=driver.SEQ_LENS)
    p.add_argument("--pred-lens", nargs="+", type=int, default=driver.PRED_LENS)
    p.add_argument("--folds", nargs="+", type=int, default=[0])
    p.add_argument("--strategies", nargs="+", default=["direct"],
                   choices=["direct", "recursive"])
    p.add_argument("--probe-batches", type=int, default=3)
    p.add_argument("--nominal-epochs", type=int, default=50)
    p.add_argument("--threshold-hours", type=float, default=12.0)
    p.add_argument("--out", type=str,
                   default=str(driver.REPO / "runs" / "bench" / "manifest.json"))
    args = p.parse_args(argv)

    cells = driver.enumerate_cells(args.tracks, args.seq_lens, args.pred_lens,
                                   args.folds, args.strategies)
    approved, rejected = [], 0
    for cell in cells:
        models = driver.models_for(cell["track"], cell["strategy"])
        try:
            cfg = _config_for(cell, models[0], args.nominal_epochs)
            bundle = DataModule(cfg).setup()  # build_model sets enc_in/dec_in/c_out later
            steps = len(bundle.train_loader)
        except Exception as e:  # data/config setup failed: skip whole cell
            print(f"  {cell['track']} seq{cell['seq_len']} "
                  f"pred{cell['pred_len']}: cell setup failed ({e}) — "
                  f"all models rejected")
            rejected += len(models)
            continue
        for model in models:
            try:
                mcfg = _config_for(cell, model, args.nominal_epochs)
                net = build_model(model, mcfg, bundle.input_size,
                                  bundle.target_indices, strategy=cell["strategy"])
                sec = probe_sec_per_batch(net, bundle.train_loader, mcfg,
                                          args.probe_batches)
                proj = estimate_train_hours(sec, steps, args.nominal_epochs)
                print(f"  {cell['track']} seq{cell['seq_len']} "
                      f"pred{cell['pred_len']} {model}: "
                      f"{sec:.3f}s/batch, ~{proj:.1f}h @ {args.nominal_epochs}ep")
                if decide(proj, args.threshold_hours,
                          _default_prompt(cell, model)):
                    approved.append({**cell, "model": model,
                                     "projected_hours": round(proj, 2)})
                else:
                    rejected += 1
            except Exception as e:  # bad model/probe: reject, keep going
                print(f"  {cell['track']} {model}: probe failed ({e}) — rejected")
                rejected += 1
            finally:
                # Release the probed model (and any CUDA-resident copy of it)
                # before moving to the next model, otherwise GPU memory
                # accumulates across probes and can OOM later in the run.
                if "net" in locals():
                    del net
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    params = {"nominal_epochs": args.nominal_epochs,
              "threshold_hours": args.threshold_hours,
              "probe_batches": args.probe_batches,
              "seq_lens": args.seq_lens, "pred_lens": args.pred_lens,
              "folds": args.folds, "strategies": args.strategies}
    write_manifest(approved, params, args.out)
    print(f"\nPre-test done: {len(approved)} approved, {rejected} rejected. "
          f"Manifest → {args.out}")


if __name__ == "__main__":
    main()
