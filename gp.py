import random
import copy
import math
import statistics
import time
import pandas as pd
from Tree import Tree, random_tree, print_equation_string

def load_data(filename, test_size=5000):
    # read the csv into a table, skipping the header row
    # (dataset2's header "f(x1,x2,x3)" has commas in it, which would create fake extra columns)
    data = pd.read_csv(filename, header=None, skiprows=1)

    # turn each row into (inputs, answer): last column is the answer, the rest are inputs
    rows = []
    for row in data.values.tolist():
        rows.append((row[:-1], row[-1]))

    # shuffle so the test set isn't just one end of the file
    random.shuffle(rows)

    # first test_size rows are the test set, the rest are training (20,000 / 5,000)
    # GP never sees the test set; it's only used at the end to judge the final tree
    test = rows[:test_size]
    train = rows[test_size:]
    return train, test

def evaluate(tree, rows):
    """
    Run the tree on every row. Returns (mean squared error, fraction of rows that needed a repair).
    The repair count lives on a fresh notepad made here, not on the tree, so it starts at 0 every time.
    """
    notepad = [0]
    total = 0
    repaired_rows = 0
    try:
        for inputs, answer in rows:
            before = notepad[0]
            guess = tree.calculate_value(inputs, notepad)
            total += (guess - answer) ** 2
            if notepad[0] > before:     # this row needed at least one protected operation
                repaired_rows += 1
    except (ValueError, OverflowError, ZeroDivisionError):
        return float('inf'), 1.0        # tree blew up (e.g. sin(inf)): worst possible error
    return total / len(rows), repaired_rows / len(rows)

def fitness(tree, rows, weight=0.001, repair_weight=0):
    """
    How wrong this tree is on the given rows (lower is better, like tour length).

    fitness = normalized MSE + weight * tree size + repair_weight * fraction of rows repaired
      normalized MSE: 0 = perfect, 1 = no better than always guessing the average answer
      repair_weight: 0 = ignore repairs (datasets 1, 2); > 0 punishes trees that lean on
                     protected log / exp / division instead of real math (dataset 3)
    """
    # average answer and how spread out the answers are (variance)
    answers = []
    for inputs, answer in rows:
        answers.append(answer)
    mean = sum(answers) / len(answers)
    variance = 0
    for answer in answers:
        variance += (answer - mean) ** 2
    variance = variance / len(answers)
    if variance == 0:               # every answer is the same: avoid dividing by zero
        variance = 1

    mse, repair_rate = evaluate(tree, rows)

    # dividing by variance puts every dataset on the same scale, so one size penalty works for all
    # the size penalty is the regularization: bigger trees cost more, which fights bloat and overfitting
    score = mse / variance + weight * len(tree.to_list()) + repair_weight * repair_rate
    if not math.isfinite(score):    # inf or nan: worst possible score
        return float('inf')
    return score

def fitness_to_weights(scores, method="power", p=0.5):
    """
    Turn fitness scores (lower is better) into selection weights (higher is better),
    so parents can be picked with random.choices(population, weights) like in TSP.

      "power": weight = (1 / score)^p  -> gaps matter; p sets how much
               (p = 0: everyone equal, p = 1: plain inverse, p = 0.5: gaps softened)
      "rank":  best tree gets weight N, next N-1, ..., worst 1  -> only the order matters
    """
    if method == "rank":
        order = sorted(range(len(scores)), key=lambda i: scores[i])   # indexes, best first
        weights = [0] * len(scores)
        for rank, i in enumerate(order):
            weights[i] = len(scores) - rank
        return weights

    weights = []
    for s in scores:
        if not math.isfinite(s):
            weights.append(0)                           # blown-up trees never get picked
        else:
            weights.append((1 / max(s, 1e-12)) ** p)    # max(...) avoids dividing by 0
    if sum(weights) == 0:                               # every tree blew up: pick evenly
        weights = [1] * len(scores)
    return weights

def tune_constants(tree, score, step=1.0, min_step=0.001, max_rounds=200):
    """
    Return a *new* Tree with the same shape, but with its constants nudged
    to lower score(tree).

    score: any function that takes a tree and returns a number (lower is better)

    DO NOT MODIFY tree!
    """
    child = copy.deepcopy(tree)

    # constants are the leaves holding a number (variables and operators are strings)
    constants = []
    for node in child.to_list():
        if not isinstance(node.data, str):
            constants.append(node)

    best = score(child)
    s = step
    rounds = 0
    # the "binary search" idea: nudge each constant a bit up and a bit down,
    # keep any move that helps; if a whole round helps nothing, halve the step
    while s > min_step and rounds < max_rounds:
        rounds += 1
        improved = False
        for node in constants:
            original = node.data
            moved = False
            for candidate in (original + s, original - s):
                node.data = candidate
                new = score(child)
                if new < best:          # keep the move
                    best = new
                    moved = True
                    improved = True
                    break
            if not moved:               # neither direction helped: undo
                node.data = original
        if not improved:                # nothing helped this round: smaller steps
            s = s / 2
    return child

def seed_trees(rows, variables, top=5, weight=0.001, repair_weight=0):
    """
    Starter trees for generation 0, picked by correlation (guides GP, doesn't replace it).

    For each piece g (x, x^2, x^3, 1/x, sin x, log x, e^x) of each variable, check how well
    it lines up with the answers. Fit the best ones as a * g(x - h) + k, then tune a, h, k.
    """
    def node(data, left=None, right=None):
        t = Tree(data)
        t.left = left
        t.right = right
        return t

    def x(v):
        return node("-", node(v), node(0.0))    # (v - h): horizontal shift h starts at 0

    answers = [answer for inputs, answer in rows]
    scored = []
    for v in variables:
        pieces = [x(v), node("*", x(v), x(v)), node("*", node("*", x(v), x(v)), x(v)),
                  node("/", node(1.0), x(v)), node("sin", x(v)), node("log", x(v)), node("^", x(v))]
        for g in pieces:
            try:
                gs = [g.calculate_value(inputs) for inputs, answer in rows]
                r = statistics.correlation(gs, answers)           # +1/-1 = lines up perfectly, 0 = not at all
                a, k = statistics.linear_regression(gs, answers)  # best stretch (a) and vertical shift (k)
            except (ValueError, OverflowError, ZeroDivisionError, statistics.StatisticsError):
                continue                                          # piece blows up or is flat on this data
            if math.isfinite(r):
                scored.append((abs(r), a, g, k))

    scored.sort(key=lambda s: s[0], reverse=True)
    seeds = []
    for r, a, g, k in scored[:top]:
        tree = node("+", node("*", node(a), g), node(k))          # a * g(x - h) + k
        seeds.append(tune_constants(tree, lambda t: fitness(t, rows, weight, repair_weight)))
    return seeds

def run_gp(train, variables, unary=False, pop_size=200, generations=50, sample_size=500,
           mutation_prob=0.1, crossover_prob=0.6, method="power", p=0.5,
           weight=0.001, repair_weight=0, n_seeds=5, max_depth=10, verbose=True):
    """
    One full GP run (same shape as the TSP loop). Uses only the training rows.
    Returns the best tree and a history (one entry per generation) for plotting.
    """
    multi = len(variables) > 1
    start = time.time()                     # clock starts before seeding, so seeding time counts

    # generation 0: a few correlation seeds + random trees for the rest
    rows = random.sample(train, sample_size)
    population = seed_trees(rows, variables, top=n_seeds, weight=weight, repair_weight=repair_weight) if n_seeds > 0 else []
    while len(population) < pop_size:
        population.append(random_tree(Unary=unary, Multi=multi))

    history = []
    for gen in range(generations):
        # fresh sample each generation; every tree in this generation is scored on the same rows
        rows = random.sample(train, sample_size)
        scores = [fitness(t, rows, weight, repair_weight) for t in population]
        best_i = min(range(pop_size), key=lambda i: scores[i])

        # track progress and diversity (share of trees that are unique formulas)
        finite = [s for s in scores if math.isfinite(s)]
        unique = len(set(str(t.print_equation()) for t in population))
        history.append({"best": scores[best_i],
                        "mean": sum(finite) / len(finite) if finite else float('inf'),
                        "diversity": unique / pop_size,
                        "size": len(population[best_i].to_list()),
                        "tree": population[best_i],         # best tree this generation (for plotting)
                        "time": time.time() - start,       # seconds since the run started
                        "repairs": evaluate(population[best_i], rows)[1]})   # share of rows the best tree needed patched
        if verbose and (gen % 10 == 0 or gen == generations - 1):
            h = history[-1]
            print(f"  gen {gen:3d}  best {h['best']:.4f}  mean {h['mean']:.3f}  "
                  f"diversity {h['diversity']:.0%}  best size {h['size']}")

        # next generation: elitism (best tree survives), then mutation / crossover / cloning
        weights = fitness_to_weights(scores, method, p)
        next_gen = [population[best_i]]              # elitism: the best score can never get worse
        while len(next_gen) < pop_size:
            parent = random.choices(population, weights)[0]
            flip = random.random()
            if flip < mutation_prob:
                child = parent.mutate(Multi=multi)
            elif flip < mutation_prob + crossover_prob:
                other = random.choices(population, weights)[0]
                child = parent.crossover(other)
            else:
                child = parent                       # clone (safe: trees are never modified)
            if child.height() > max_depth:           # too deep: keep the parent instead (second bloat control)
                child = parent
            next_gen.append(child)
        population = next_gen

    # final pick: score the last generation on a bigger training sample
    rows = random.sample(train, min(2000, len(train)))
    best = min(population, key=lambda t: fitness(t, rows, weight, repair_weight))
    return best, history

def main():
    train, test = load_data("data/dataset1.csv")
    best, history = run_gp(train, ["x1"])
    print("best tree:", print_equation_string(best.print_equation()))
    print("test error (normalized MSE, no size penalty):", fitness(best, test, weight=0))

if __name__ == "__main__":
    main()
