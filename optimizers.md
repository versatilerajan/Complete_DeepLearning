# Optimizers, Part 1: Why Plain Gradient Descent Isn't Enough

*Companion notes introducing optimizers — the algorithms that decide how a network's weights actually get updated — recapping the three gradient descent variants, and diagnosing three real limitations of plain gradient descent (learning rate sensitivity, one-size-fits-all step sizes, and getting stuck at local minima and saddle points), each demonstrated with a real computed experiment.*

---

## Table of Contents

1. [What Is an Optimizer?](#1-what-is-an-optimizer)
2. [Recap: The Three Gradient Descent Variants](#2-recap-the-three-gradient-descent-variants)
3. [Challenge 1: Learning Rate Sensitivity](#3-challenge-1-learning-rate-sensitivity)
4. [Challenge 2: One Learning Rate for Every Parameter](#4-challenge-2-one-learning-rate-for-every-parameter)
5. [Real Experiment: The Ravine Problem](#5-real-experiment-the-ravine-problem)
6. [Challenge 3: Local Minima and Saddle Points](#6-challenge-3-local-minima-and-saddle-points)
7. [Real Experiment: Getting Stuck, Two Ways](#7-real-experiment-getting-stuck-two-ways)
8. [Coming Up in This Series](#8-coming-up-in-this-series)
9. [Code: Reproducing Every Experiment](#9-code-reproducing-every-experiment)
10. [Key Takeaways](#10-key-takeaways)
11. [Further Reading](#11-further-reading)

---

## 1. What Is an Optimizer?

Every document in this series so far — [weight initialization](weight-initialization.md), [batch normalization](batch-normalization.md), [activation functions](activation-functions.md) — has been about making sure a good gradient signal actually reaches every layer. An **optimizer** is the final piece: given that gradient, it's the algorithm that decides exactly how to turn it into a parameter update.

![What is an optimizer, exactly?](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/role_of_optimizer.png)

The simplest possible optimizer is plain **Gradient Descent** — covered in full in the [gradient descent notes](gradient-descent.md) — which just takes a fixed-size step in the direction of steepest descent every update. It works, but as this document covers, it leaves real performance and reliability on the table.

---

## 2. Recap: The Three Gradient Descent Variants

As a quick reminder from the [gradient descent notes](gradient-descent.md):

- **Batch Gradient Descent** — one update per epoch, using the entire dataset's gradient.
- **Stochastic Gradient Descent (SGD)** — one update per training example.
- **Mini-Batch Gradient Descent** — one update per small batch; the practical default.

All three use the *exact same* update rule, `w ← w − η·∂L/∂w` — they only differ in how much data goes into computing the gradient before applying it. Every challenge in this document applies equally to all three variants, since it's about the update rule itself, not how the gradient was computed.

---

## 3. Challenge 1: Learning Rate Sensitivity

The learning rate `η` has to be chosen just right: too small and training crawls forward almost imperceptibly; too large and updates overshoot the minimum, potentially diverging entirely (see the concrete numerical examples of both failure modes in the [gradient descent](gradient-descent.md) and [weight initialization](weight-initialization.md) notes, where overly large updates were shown driving loss to NaN within a handful of epochs). There's no single "safe" value that works across every problem, every architecture, and every stage of training — which is itself a real practical burden.

---

## 4. Challenge 2: One Learning Rate for Every Parameter

Plain gradient descent applies the **exact same** learning rate to every single weight in the network, regardless of how that particular weight's loss landscape behaves. But different weights can have wildly different sensitivities — the loss might change very sharply along one weight's direction and very gradually along another's. A single shared learning rate has no way to accommodate both at once: whatever value works for the sensitive direction will be far too small for the insensitive one, and vice versa.

---

## 5. Real Experiment: The Ravine Problem

This exact scenario — one direction far steeper than another — has a name: the **ravine problem**. Here's a real, simple loss surface (a bowl-shaped function, steep along one axis and shallow along the other) with plain gradient descent run at the largest learning rate that doesn't diverge in the steep direction:

![Real experiment: the ravine problem](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/ravine_problem_real.png)

*The chosen learning rate is exactly at the edge of what the steep direction can tolerate — any larger and it would diverge. But that same learning rate makes only glacial progress along the shallow direction: after 30 steps, the point has zigzagged its way to a good position in the steep direction, but has barely covered a fraction of the distance needed in the shallow direction. This single experiment captures Challenge 2 concretely: there is no learning rate that's simultaneously "just right" for both directions.*

---

## 6. Challenge 3: Local Minima and Saddle Points

Real loss landscapes, especially in high-dimensional neural networks, are rarely a single smooth bowl. Two specific shapes cause trouble for plain gradient descent:

- **Local minima** — a dip that's better than its immediate surroundings, but not the best dip available anywhere in the landscape (the *global* minimum). Gradient descent has no way to "see" past the walls of whatever valley it's currently in — once it settles into a local minimum, the gradient there is zero, and it has no mechanism to escape and search elsewhere.
- **Saddle points** — a point where the landscape curves *up* in one direction and *down* in another simultaneously. The gradient at the saddle point itself is exactly zero (just like at a true minimum), and gradients *near* the saddle are very small in every direction — causing training to slow to a crawl even though it isn't actually stuck at a minimum.

---

## 7. Real Experiment: Getting Stuck, Two Ways

![Real experiment: local minima and saddle points](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/local_minima_saddle.png)

*Left: starting gradient descent in the basin of a shallower local minimum, it converges there and stops permanently — the true (deeper, better) global minimum sits in a completely different basin that gradient descent has no way to discover from here. Right: on a genuine saddle-point landscape (`f(x,y) = x² − y²`), starting very close to the saddle along the unstable direction, the point spends roughly 140 update steps crawling almost imperceptibly close to the saddle — the gradient there is nearly zero — before finally escaping and accelerating rapidly away. Nearly 75% of the steps shown are spent in that slow plateau.*

---

## 8. Coming Up in This Series

Each of the challenges above has a specific, well-established fix, covered in upcoming documents in this series:

![Coming up in this series](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/optimizers_roadmap.png)

- **Momentum** smooths out the ravine zigzag by accumulating a running average of recent gradients, rather than reacting fully to just the current one.
- **Nesterov** improves on Momentum by computing the gradient at a "look-ahead" point rather than the current position.
- **Adagrad** directly addresses Challenge 2 by giving every individual parameter its own adaptive learning rate, based on how large that parameter's past gradients have been.
- **RMSprop** fixes a specific weakness in Adagrad, where the per-parameter learning rate shrinks so aggressively over time that learning can grind to a halt too early.
- **Adam** combines the ideas behind Momentum and RMSprop into a single optimizer, and is by far the most widely used choice in modern deep learning.

---

## 9. Code: Reproducing Every Experiment

```python
import numpy as np

# ---------------------------------------------------------
# The ravine problem (Section 5)
# ---------------------------------------------------------
a, b = 0.5, 10.0  # curvature: shallow in x, steep in y

def ravine_grad(x, y):
    return 2 * a * x, 2 * b * y

def run_gd_ravine(lr, start=(-4.0, 1.0), steps=50):
    x, y = start
    path = [(x, y)]
    for _ in range(steps):
        gx, gy = ravine_grad(x, y)
        x -= lr * gx
        y -= lr * gy
        path.append((x, y))
    return np.array(path)

# The largest lr that stays stable in the steep (y) direction is just under 1/b = 0.1
path = run_gd_ravine(lr=0.09)
print("After 50 steps: x =", path[-1, 0], " y =", path[-1, 1])
print("Note how far x still is from 0, despite y having converged.")

# ---------------------------------------------------------
# Local minima trap (Section 7, left panel)
# ---------------------------------------------------------
def landscape(w):
    return 0.04*w**2 - 1.3*np.exp(-(w-2.0)**2/0.5) - 1.7*np.exp(-(w+0.8)**2/0.5)

def landscape_grad(w, h=1e-4):
    return (landscape(w + h) - landscape(w - h)) / (2 * h)

w = 2.35  # starts in the shallower local minimum's basin
for _ in range(60):
    w -= 0.1 * landscape_grad(w)
print(f"\nStuck at w={w:.3f}, loss={landscape(w):.3f} (global min is near w=-0.8, loss={landscape(-0.8):.3f})")

# ---------------------------------------------------------
# Saddle point plateau (Section 7, right panel)
# ---------------------------------------------------------
def saddle_grad(x, y):
    return 2 * x, -2 * y

x, y = 3.0, 1e-6  # start very close to the saddle along the unstable direction
saddle_path_y = [y]
for _ in range(190):
    gx, gy = saddle_grad(x, y)
    x -= 0.05 * gx
    y -= 0.05 * gy
    saddle_path_y.append(y)

escape_step = np.argmax(np.abs(saddle_path_y) > 0.5)
print(f"\nSpent {escape_step} steps within the saddle's plateau before escaping.")
```

---

## 10. Key Takeaways

- An optimizer is the algorithm that turns a computed gradient into an actual parameter update — plain Gradient Descent (in any of its Batch/SGD/Mini-batch forms) is the simplest possible choice, but has real, demonstrable limitations.
- **Learning rate sensitivity**: too small stalls progress, too large causes divergence, and there's no universally correct value.
- **One learning rate for every parameter** breaks down whenever the loss landscape has very different curvature in different directions — a real experiment shows this "ravine problem" directly: the only stable learning rate for the steep direction makes the shallow direction crawl.
- **Local minima** trap gradient descent permanently in a suboptimal valley, with zero mechanism to search elsewhere; a real experiment shows exactly this happening.
- **Saddle points** cause a long, painful plateau (measured directly: ~140 steps of near-zero movement) before gradient descent eventually escapes along the unstable direction.
- Every one of these challenges has a specific, standard fix — Momentum, Nesterov, Adagrad, RMSprop, and Adam — each covered in its own document later in this series.

---

## 11. Further Reading

- Ruder, S. (2016). *An Overview of Gradient Descent Optimization Algorithms* — a widely-cited survey covering exactly the algorithms previewed in Section 8, in more mathematical depth.
- Dauphin, Y. et al. (2014). *Identifying and Attacking the Saddle Point Problem in High-Dimensional Non-Convex Optimization* — argues that saddle points, not local minima, are the dominant obstacle in high-dimensional neural network loss landscapes.
- See also this repo's [`gradient-descent.md`](gradient-descent.md) (the three GD variants this document builds on) and [`backprop-part1.md`](backprop-part1.md) / [`backprop-part2.md`](backprop-part2.md) (how the gradient being optimized here is actually computed).

---

*Diagrams in this document were generated programmatically — including real gradient descent runs on a ravine-shaped loss surface, a local-minima landscape, and a genuine saddle point — and are hosted in this repo's [`images/`](https://github.com/versatilerajan/deepcontent/tree/main/images) folder.*
