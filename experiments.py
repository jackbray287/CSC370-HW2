import os
import sys
import csv
import random
import statistics
import matplotlib
matplotlib.use("Agg")                # save plots to files instead of opening windows
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from Tree import VARS, print_equation_string
from gp import load_data, fitness, run_gp

OUT_DIR = "experiments"

# dataset number -> (file, variables, use unary functions?, repair penalty weight)
# repair weight is 0 where repairs are rare (only + - * /); dataset 3's log/exp get patched a lot
DATASETS = {
    "1": ("data/dataset1.csv", ["x1"], False, 0),
    "2": ("data/dataset2.csv", VARS, False, 0),
    "3": ("data/dataset3.csv", ["x1"], True, 0.1),      # -> KNOB
}

def compare_seeding(name, runs=5, generations=50, seed=0, threshold=0.02):
    """
    Run GP `runs` times with correlation seeds and `runs` times without,
    on the same train/test split. Saves plots and a CSV to experiments/.

    threshold: "converged" means the best fitness reached this value (same target for both modes)
    """
    filename, variables, unary, repair_weight = DATASETS[name]
    random.seed(seed)                            # same seed -> same split and same results each time
    train, test = load_data(filename)

    results = {"seeded": [], "unseeded": []}
    for label, n_seeds in (("seeded", 5), ("unseeded", 0)):
        for r in range(runs):
            best, history = run_gp(train, variables, unary=unary, generations=generations,
                                   n_seeds=n_seeds, repair_weight=repair_weight, verbose=False)
            test_error = fitness(best, test, weight=0)    # test set: only used here, after training
            formula = print_equation_string(best.print_equation())
            gens, secs = time_to_threshold(history, threshold)
            results[label].append({"history": history, "test_error": test_error, "formula": formula,
                                   "gens_to_threshold": gens, "secs_to_threshold": secs})
            reached = f"reached {threshold} at gen {gens} ({secs:.1f}s)" if gens is not None else f"never reached {threshold}"
            print(f"  {label:8s} run {r + 1}/{runs}  final best {history[-1]['best']:.4f}  "
                  f"test error {test_error:.6f}  {reached}  {formula}")

    os.makedirs(OUT_DIR, exist_ok=True)
    save_plot(name, results, generations)
    save_convergence(name, results, threshold)
    save_csv(name, results)

    # curves over the data only make sense with one input variable (datasets 1 and 3)
    if len(variables) == 1:
        for label, runs in results.items():
            best_run = min(runs, key=lambda run: run["test_error"])
            save_evolution(name, label, train, best_run["history"])

def time_to_threshold(history, threshold):
    # first generation whose best fitness is at or below the threshold, and the seconds it took
    for g, h in enumerate(history):
        if h["best"] <= threshold:
            return g, h["time"]
    return None, None                            # never converged in this run

def save_convergence(name, results, threshold):
    """
    Left: best fitness vs. wall-clock seconds (includes seeding time).
    Right: seconds to reach the threshold, one dot per run, bar = median.
    """
    colors = {"seeded": "tab:blue", "unseeded": "tab:orange"}
    fig, (ax_time, ax_bar) = plt.subplots(1, 2, figsize=(12, 4.5))

    for label, runs in results.items():
        for i, run in enumerate(runs):
            ax_time.plot([h["time"] for h in run["history"]], [h["best"] for h in run["history"]],
                         color=colors[label], alpha=0.6, label=label if i == 0 else None)
    ax_time.axhline(threshold, color="gray", linestyle="--", label=f"threshold {threshold}")
    ax_time.set_yscale("log")
    ax_time.set_title(f"Dataset {name}: best fitness over time")
    ax_time.set_xlabel("seconds since run started (includes seeding)")
    ax_time.set_ylabel("best fitness (lower is better)")
    ax_time.legend()

    labels = list(results)
    for i, label in enumerate(labels):
        secs = [run["secs_to_threshold"] for run in results[label] if run["secs_to_threshold"] is not None]
        missed = len(results[label]) - len(secs)
        if secs:
            ax_bar.bar(i, statistics.median(secs), color=colors[label], alpha=0.4)
            ax_bar.scatter([i] * len(secs), secs, color=colors[label], zorder=3)
        ax_bar.text(i, 0, f"{len(secs)}/{len(results[label])} converged" + (f"\n({missed} never)" if missed else ""),
                    ha="center", va="bottom", fontsize=9)
    ax_bar.set_xticks(range(len(labels)))
    ax_bar.set_xticklabels(labels)
    ax_bar.set_title(f"Dataset {name}: seconds to reach fitness {threshold}")
    ax_bar.set_ylabel("seconds (bar = median, dots = runs)")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"dataset{name}_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")

def curve(tree, xs):
    # the tree's output at each x (None where it blows up, so matplotlib leaves a gap)
    ys = []
    for x in xs:
        try:
            ys.append(tree.calculate_value([x]))
        except (ValueError, OverflowError, ZeroDivisionError):
            ys.append(None)
    return ys

def save_evolution(name, label, train, history):
    """
    The data points with the best tree's curve drawn on top, as it changes over generations.
    Saves a snapshot plot (.png) and an animation (.gif) for one run.
    """
    points = random.sample(train, 1000)
    data_x = [inputs[0] for inputs, answer in points]
    data_y = [answer for inputs, answer in points]
    lo, hi = min(data_x), max(data_x)
    xs = [lo + (hi - lo) * i / 300 for i in range(301)]
    margin = (max(data_y) - min(data_y)) * 0.2
    ylim = (min(data_y) - margin, max(data_y) + margin)   # keep wild curves from squashing the data
    last = len(history) - 1

    # snapshot plot: a few generations on one chart, colored early (dark) to late (yellow)
    snapshots = sorted(set(g for g in (0, 2, 5, 10, 20, last) if g <= last))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(data_x, data_y, s=4, color="lightgray", label="data")
    for i, g in enumerate(snapshots):
        color = plt.cm.viridis(i / max(len(snapshots) - 1, 1))
        ax.plot(xs, curve(history[g]["tree"], xs), color=color, linewidth=2,
                label=f"gen {g} (fitness {history[g]['best']:.4f})")
    ax.set_ylim(ylim)
    ax.set_title(f"Dataset {name} ({label}): best tree over generations")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"dataset{name}_{label}_evolution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")

    # animation: one frame per generation
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(data_x, data_y, s=4, color="lightgray")
    line, = ax.plot([], [], color="tab:red", linewidth=2.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")

    def draw(g):
        line.set_data(xs, curve(history[g]["tree"], xs))
        ax.set_title(f"Dataset {name} ({label}): generation {g}, best fitness {history[g]['best']:.4f}")
        return line,

    anim = FuncAnimation(fig, draw, frames=len(history))
    path = os.path.join(OUT_DIR, f"dataset{name}_{label}_evolution.gif")
    anim.save(path, writer=PillowWriter(fps=5))
    plt.close(fig)
    print(f"saved {path}")

def save_plot(name, results, generations):
    # left: best fitness per generation, right: diversity per generation
    # faint lines = individual runs, bold line = median across runs
    colors = {"seeded": "tab:blue", "unseeded": "tab:orange"}
    fig, (ax_fit, ax_div) = plt.subplots(1, 2, figsize=(12, 4.5))
    gens = range(generations)

    for label, runs in results.items():
        for key, ax in (("best", ax_fit), ("diversity", ax_div)):
            for run in runs:
                ax.plot(gens, [h[key] for h in run["history"]], color=colors[label], alpha=0.2)
            median = [statistics.median(run["history"][g][key] for run in runs) for g in gens]
            ax.plot(gens, median, color=colors[label], linewidth=2.5, label=f"{label} (median)")

    ax_fit.set_yscale("log")                     # scores shrink by orders of magnitude, so log scale
    ax_fit.set_title(f"Dataset {name}: best fitness per generation")
    ax_fit.set_xlabel("generation")
    ax_fit.set_ylabel("best fitness (lower is better)")
    ax_fit.legend()

    ax_div.set_title(f"Dataset {name}: diversity per generation")
    ax_div.set_xlabel("generation")
    ax_div.set_ylabel("share of unique trees")
    ax_div.set_ylim(0, 1)
    ax_div.legend()

    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"dataset{name}_seeded_vs_unseeded.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")

def save_csv(name, results):
    # one row per run: what it found and how well it did on the test set
    path = os.path.join(OUT_DIR, f"dataset{name}_seeded_vs_unseeded.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "run", "final_best_fitness", "test_error", "final_diversity",
                         "gens_to_threshold", "secs_to_threshold", "total_secs", "final_repair_rate", "formula"])
        for label, runs in results.items():
            for r, run in enumerate(runs):
                last = run["history"][-1]
                writer.writerow([label, r + 1, last["best"], run["test_error"], last["diversity"],
                                 run["gens_to_threshold"], run["secs_to_threshold"], last["time"],
                                 last["repairs"], run["formula"]])
    print(f"saved {path}")

def main():
    # usage: python experiments.py 1      (or 2, 3; several at once: python experiments.py 1 2)
    names = sys.argv[1:] or ["1"]
    for name in names:
        print(f"===== dataset {name}: seeded vs unseeded =====")
        compare_seeding(name)

if __name__ == "__main__":
    main()
