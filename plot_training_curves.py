"""Plot TensorBoard scalars to PNG files.

Reads the event files written by an RL training run and renders each scalar to
a PNG with matplotlib. This exists because the interactive TensorBoard UI (and
on-screen rendering generally) is unavailable on this machine -- writing images
to disk is the headless path to reading a run's results.

Usage
-----
    python plot_training_curves.py <logdir>
    python plot_training_curves.py <logdir> --out plots --smooth 0.9
    python plot_training_curves.py <logdir> --list

`logdir` may point either at a single run directory (one containing
`events.out.tfevents.*`) or at a parent holding several runs, in which case
every run found underneath is plotted and runs are overlaid on shared axes.

Requires `tensorboard` and `matplotlib`:

    pip install tensorboard matplotlib
"""

import argparse
import os
import sys

# Use a non-interactive backend: this script is meant to run headless (in the
# training container / WSL), where no display is available.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    sys.exit(
        "tensorboard is required to read event files.\n"
        "    pip install tensorboard matplotlib"
    )


def find_runs(logdir):
    """Return (run_name, directory) for every directory holding event files.

    A run is any directory directly containing an `events.out.tfevents.*` file.
    """
    runs = []
    for dirpath, _dirnames, filenames in os.walk(logdir):
        if any(f.startswith("events.out.tfevents") for f in filenames):
            name = os.path.relpath(dirpath, logdir)
            runs.append(("." if name == os.curdir else name, dirpath))
    return sorted(runs)


def load_scalars(run_dir):
    """Load every scalar series from one run directory.

    Returns {tag: (steps, values)}. `size_guidance=0` means "load all points"
    rather than the default downsampled subset, so the curve is not silently
    thinned out.
    """
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()

    series = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        series[tag] = ([e.step for e in events], [e.value for e in events])
    return series


def smooth(values, weight):
    """Exponential moving average, matching TensorBoard's smoothing slider.

    `weight` is in [0, 1); 0 disables smoothing. RL reward curves are noisy
    enough that the unsmoothed line often hides the trend.
    """
    if weight <= 0 or not values:
        return values

    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return smoothed


def safe_filename(tag):
    """Turn a scalar tag such as `Train/mean_reward` into a usable filename."""
    return tag.replace("/", "_").replace("\\", "_").replace(" ", "_")


def plot_tag(tag, per_run, out_dir, weight):
    """Render one scalar tag, overlaying every run that reports it."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for run_name, (steps, values) in sorted(per_run.items()):
        line = ax.plot(steps, smooth(values, weight), label=run_name)[0]
        if weight > 0:
            # Show the raw series faintly behind the smoothed line so the
            # actual variance stays visible.
            ax.plot(steps, values, alpha=0.15, color=line.get_color())

    ax.set_title(tag)
    ax.set_xlabel("step")
    ax.set_ylabel(tag.split("/")[-1])
    ax.grid(alpha=0.3)
    if len(per_run) > 1:
        ax.legend()
    fig.tight_layout()

    path = os.path.join(out_dir, safe_filename(tag) + ".png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def summarize(tag, per_run):
    """Print first/last/max for a tag so the trend is readable without images."""
    for run_name, (steps, values) in sorted(per_run.items()):
        if not values:
            continue
        print(
            f"  {tag:<40} {run_name:<12} "
            f"first={values[0]:>12.4f}  last={values[-1]:>12.4f}  "
            f"max={max(values):>12.4f}  points={len(values)}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot TensorBoard scalars to PNG files (headless)."
    )
    parser.add_argument("logdir", help="run directory, or a parent containing runs")
    parser.add_argument(
        "--out", default="plots", help="output directory for PNGs (default: plots)"
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.6,
        help="EMA smoothing weight in [0, 1); 0 disables (default: 0.6)",
    )
    parser.add_argument(
        "--tags", nargs="*", help="only plot these tags (default: all scalars)"
    )
    parser.add_argument(
        "--list", action="store_true", help="list available tags and exit"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.logdir):
        sys.exit(f"not a directory: {args.logdir}")
    if not 0 <= args.smooth < 1:
        sys.exit(f"--smooth must be in [0, 1), got {args.smooth}")

    runs = find_runs(args.logdir)
    if not runs:
        sys.exit(f"no event files found under {args.logdir}")

    print(f"found {len(runs)} run(s) under {args.logdir}")

    # tag -> {run_name: (steps, values)}
    by_tag = {}
    for run_name, run_dir in runs:
        scalars = load_scalars(run_dir)
        print(f"  {run_name}: {len(scalars)} scalar tag(s)")
        for tag, data in scalars.items():
            by_tag.setdefault(tag, {})[run_name] = data

    if args.list:
        print("\navailable tags:")
        for tag in sorted(by_tag):
            print(f"  {tag}")
        return

    if args.tags:
        missing = [t for t in args.tags if t not in by_tag]
        for t in missing:
            print(f"warning: tag not found, skipping: {t}", file=sys.stderr)
        by_tag = {t: v for t, v in by_tag.items() if t in args.tags}
        if not by_tag:
            sys.exit("none of the requested tags were found; try --list")

    os.makedirs(args.out, exist_ok=True)

    print("\nsummary:")
    for tag in sorted(by_tag):
        summarize(tag, by_tag[tag])

    print()
    for tag in sorted(by_tag):
        path = plot_tag(tag, by_tag[tag], args.out, args.smooth)
        print(f"wrote {path}")

    print(f"\n{len(by_tag)} plot(s) written to {args.out}/")


if __name__ == "__main__":
    main()
