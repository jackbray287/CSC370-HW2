# Symbolic Regression with Genetic Programming

CSC 370 (Fall 2026), Homework 2. We use genetic programming (GP) to recover the hidden formula behind three datasets of 25,000 points each. Every candidate formula is an expression tree, and the population evolves through selection, crossover, and mutation. The approach follows John Koza's [quadratic example](http://www.genetic-programming.com/gpquadraticexample.html), with additions for seeding, constant tuning, and bloat control.

## Results

| Dataset | Inputs | Building blocks | Best formula found | Our read on f |
|---|---|---|---|---|
| 1 | x | + − × ÷, integer constants | x² − 6.0x + 13.99 | **f(x) = x² − 6x + 14 = (x − 3)² + 5** |
| 2 | x1, x2, x3 | + − × ÷, real constants | 6.93 × 10⁻¹¹ · x1 · x2 / x3² | **f = G · x1 · x2 / x3²**, i.e. Newton's law of gravitation (G ≈ 6.67 × 10⁻¹¹) |
| 3 (extra credit) | x | adds sin, log, eˣ | no clean closed form | Best fits combine polynomial terms with sin(~2x), plus e⁻ˣ · sin terms in some runs; test error ≈ 0.004 |

Each dataset got 10 runs: 5 seeded and 5 unseeded, all on the same train/test split. Test error is normalized MSE on the 5,000 held-out rows, where 0 is a perfect fit and 1 is no better than always guessing the mean. For dataset 2 it is measured on the asinh scale (see below).

| Dataset | Gens | Runs reaching fitness 0.02 (seeded / unseeded) | Best test error | Time per run |
|---|---|---|---|---|
| 1 | 100 | 5/5 at gen 0 / 5/5 by gen 11–73 | 1.9 × 10⁻⁶ | ~30–45 s |
| 2 | 300 | 1/5 / 2/5 | 0.0049 | ~2–5 min |
| 3 | 2500 | 0/5 / 0/5 | 0.0039 | ~7–40 min |

What the runs show:

- **Dataset 1 is solved reliably.** All 10 runs land on x² − 6x + 14 to within a few hundredths on each coefficient. With seeding the right shape exists at generation 0, so the remaining generations only polish the constants.
- **Dataset 2 is found, but not consistently.** Three runs found the clean x1·x2/x3² form, with coefficients between 6.59 and 6.94 × 10⁻¹¹. Other runs got stuck in larger rational expressions. Seeding did not help here, because the seed shapes (single-variable a·g(x − h) + k) can't express a product of three variables.
- **Dataset 3 is the hard one.** Seeded runs did better on average (5/5 under fitness 0.065), while two unseeded runs collapsed into a trivial log(exp(...)) form with test error ≈ 0.7. The best models fit the data well but differ in form from run to run, so we don't claim a single "true" f.

Plots and per-run CSVs are in [`experiment_results/`](experiment_results/):

- `datasetN_seeded_vs_unseeded.png`: best fitness and diversity per generation (faint lines are individual runs, bold lines are medians)
- `datasetN_convergence.png`: best fitness over wall-clock time, and seconds to reach the 0.02 threshold
- `datasetN_{seeded,unseeded}_evolution.png/.gif`: the best tree's curve drawn over the data across generations (datasets 1 and 3 only)
- `datasetN_seeded_vs_unseeded.csv`: one row per run with final fitness, test error, diversity, timing, and the simplified formula

## How it works

### Representation (`Tree.py`)

Each individual is a binary expression tree. Internal nodes are `+ − × ÷` or, with `Unary=True`, the one-child functions `sin`, `log`, and `exp` (written `^`). Leaves are variables (`x1`, `x2`, `x3`) or numeric constants. Trees are treated as immutable: `mutate` and `crossover` deep-copy the tree and return a new one, as the assignment recommends.

Protected operators keep every tree evaluable:

- `a / 0` returns `a`
- `log(v)` for `v ≤ 0` returns 0
- `exp(v)` caps `v` at 50 to prevent overflow

Each tree counts how many times a protection fired (`repairs`).

- **Random trees** use the "grow" method: start at depth 3 and stop early at a leaf 30% of the time. Leaves are a variable or a constant with equal odds.
- **Mutation** is a point mutation. An operator swaps to a different operator of the same arity, so the tree stays valid. A variable becomes a constant or another variable, and a constant becomes a variable (40%) or a new constant.
- **Crossover** replaces one child of a random internal node in parent A with a copied subtree from parent B. That subtree is an internal node 75% of the time, so crossover mostly exchanges real chunks rather than single leaves.
- **Constants** are drawn uniformly from [−5, 5]. For dataset 2 they are drawn on a log scale (±10⁻¹²…10¹), because the true constant (~6.7 × 10⁻¹¹) can never come out of a [−5, 5] draw.

### Fitness and regularization (`gp.py`)

```
fitness = MSE / Var(y) + 0.001 × (number of nodes)
```

Dividing MSE by the variance of the answers puts all three datasets on the same scale (0 is perfect, 1 matches the predict-the-mean baseline), so a single size penalty works for all of them. The node-count term is the regularization: it pushes GP toward simpler formulas and limits bloat. As a second bloat control, any child deeper than 6 is thrown out and replaced by its parent.

Dataset 2's outputs span roughly 0 to 10¹¹. On a plain scale a handful of huge rows would dominate the score, so for this dataset errors are computed on an asinh scale (`relative=True`). asinh behaves like log for large values and keeps the sign, so being off by a factor of 2 costs about the same on every row.

### Seeding and constant tuning

`seed_trees` builds 5 starter trees for generation 0, and random trees fill the rest of the population. For each variable it tries seven basic shapes: x, x², x³, 1/x, sin x, log x, and eˣ. It ranks them by how strongly they correlate with the answers, then fits the top 5 as `a · g(x − h) + k`.

`tune_constants` then refines a, h, and k by coordinate search. It nudges each constant up and down, keeps any move that improves fitness, and halves the step size when a full round helps nothing. On a log scale it multiplies and divides instead of adding and subtracting.

### The GP loop

Each generation:

1. Score every tree on a fresh random sample of 500 training rows (cheaper than scoring all 20,000, and it adds noise that discourages overfitting to particular rows).
2. Carry the best tree over unchanged (elitism).
3. Fill the rest of the population by fitness-proportional selection, with weights (1/fitness)^0.5. Each child is a mutation (10%), a crossover (60%), or a clone (30%).

After the last generation, the final tree is chosen on a larger 2,000-row training sample. The test set is used only once, after training, to report test error.

### Overfitting

- **Train/test split:** each dataset is shuffled and split into 20,000 training rows and 5,000 test rows. GP never sees the test rows.
- **Size penalty** in the fitness function, plus the depth limit of 6.
- **Resampling:** each generation is scored on a different 500-row sample.

### Parameters

| Parameter | Value |
|---|---|
| Population size | 200 |
| Generations | 100 / 300 / 500 (datasets 1 / 2 / 3) |
| Rows scored per generation | 500 (fresh sample each time) |
| Mutation / crossover / clone | 0.10 / 0.60 / 0.30 |
| Selection | fitness-proportional, weight = (1/fitness)^0.5, elitism of 1 |
| Size penalty | 0.001 per node |
| Initial depth / max depth | 3 / 6 |
| Seed trees | 5 (0 for unseeded runs) |
| Train / test split | 20,000 / 5,000 |
| Independent runs | 5 seeded + 5 unseeded per dataset |
| Convergence threshold | best fitness ≤ 0.02 |
| Random seed | 0 |

Diversity is tracked every generation as the share of trees that are unique formulas, and it is plotted in the `seeded_vs_unseeded` figures.

## Running it

Requires Python 3.10+ (for `statistics.correlation` and `linear_regression`).

```bash
pip install pandas sympy matplotlib pillow

python gp.py                  # one quick GP run on dataset 1
python experiments.py 1       # seeded vs. unseeded comparison on dataset 1
python experiments.py 1 2 3   # all three datasets (dataset 3 alone takes a few hours)
```

`experiments.py` writes its plots and CSVs to `experiments/`. The committed results were renamed to `experiment_results/`.

On a Slurm cluster, `sbatch run_experiments.sh` runs the three datasets as a parallel job array (one CPU, 32 GB, and 4 hours each). Uncomment the right `module load` line for your cluster first. Dataset 3's 10 runs took about 3.6 hours in total, so the 4-hour limit is tight.

## Repository layout

```
Tree.py               expression tree: evaluation, mutation, crossover, random trees
gp.py                 data loading, fitness, seeding, constant tuning, the GP loop
experiments.py        seeded-vs-unseeded experiments, plots, CSVs
run_experiments.sh    Slurm job array, one job per dataset
data/                 dataset1.csv, dataset2.csv, dataset3.csv
experiment_results/   plots, GIFs, and per-run CSVs from the final experiments
instructions/         assignment PDF and the Koza tutorial
tsp_ga (1).ipynb      the course's TSP genetic algorithm, which our GP loop is modeled on
```

## Known quirks

- Formulas are printed by rebuilding the tree in SymPy and expanding it. SymPy doesn't apply our protected operators, so a printed formula can contain `zoo` (complex infinity, from a division by a zero constant) or `I*pi` (from the log of a negative constant) even though the tree itself evaluates fine.
- Expanding a formula can make it look much longer than the tree really is (see some of the dataset 3 formulas).

## Authors

Jack Bray ([@jackbray287](https://github.com/jackbray287)) and Allan Engstrom
