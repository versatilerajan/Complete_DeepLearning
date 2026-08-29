# Gradient Descent: Batch, Stochastic, and Mini-Batch

*Companion notes on the three ways to run gradient descent — computing the update from the whole dataset, from one example at a time, or from small batches in between — and the practical trade-offs (speed, stability, memory, hardware alignment) that make mini-batch the default choice in deep learning.*

---

## Table of Contents

1. [Gradient Descent: The Basic Idea](#1-gradient-descent-the-basic-idea)
2. [Batch Gradient Descent](#2-batch-gradient-descent)
3. [Stochastic Gradient Descent (SGD)](#3-stochastic-gradient-descent-sgd)
4. [Mini-Batch Gradient Descent](#4-mini-batch-gradient-descent)
5. [Comparing the Three Paths](#5-comparing-the-three-paths)
6. [Convergence Speed in Practice](#6-convergence-speed-in-practice)
7. [A Hidden Benefit of Noise: Escaping Local Minima](#7-a-hidden-benefit-of-noise-escaping-local-minima)
8. [Implementation Tip: Why Batch Sizes Are Powers of 2](#8-implementation-tip-why-batch-sizes-are-powers-of-2)
9. [Summary Table](#9-summary-table)
10. [Code: All Three Variants](#10-code-all-three-variants)
11. [Key Takeaways](#11-key-takeaways)
12. [Further Reading](#12-further-reading)

---

## 1. Gradient Descent: The Basic Idea

**Gradient descent** is the optimization algorithm behind training almost every neural network (see the [backpropagation notes](backprop-part1.md) for how the gradient itself is computed via the chain rule). At every step, it updates each parameter by moving it a small amount in the direction that reduces the loss the fastest:

```
w ← w − η · ∂L/∂w
```

where `η` (eta) is the learning rate. The three variants covered here don't change this update rule at all — they only differ in **how much data is used to compute the gradient** before applying it.

---

## 2. Batch Gradient Descent

**Batch Gradient Descent** computes the gradient using the **entire training dataset** before making a single parameter update:

```
∂L/∂w = (1/m) · Σᵢ ∂Lᵢ/∂w        (averaged over all m training examples)
```

Only after this full pass does the update `w ← w − η·∂L/∂w` happen — meaning exactly **one parameter update per epoch**.

**Pros:** the gradient is an exact, stable estimate of the true direction of steepest descent, and the computation vectorizes cleanly across the whole dataset.
**Cons:** for a large dataset, computing one gradient requires processing every single row — expensive, and painfully slow to see any progress, since nothing updates until the entire dataset has been swept through.

---

## 3. Stochastic Gradient Descent (SGD)

**Stochastic Gradient Descent** goes to the opposite extreme: compute the gradient from just **one training example**, update immediately, then move to the next example:

```
∂L/∂w ≈ ∂Lᵢ/∂w        (a single example's gradient, used as a noisy estimate of the true gradient)
```

This means **one parameter update per training example** — for a dataset of `m` rows, that's `m` updates per epoch, compared to batch gradient descent's single update.

**Pros:** updates happen far more frequently, so the model starts improving almost immediately rather than waiting for a full pass over the data; the added noise can even help the model escape poor local minima (see Section 7).
**Cons:** since every update is based on just one example, the path toward the minimum is noisy and erratic rather than smooth.

---

## 4. Mini-Batch Gradient Descent

**Mini-Batch Gradient Descent** is the middle ground: split the dataset into small batches (commonly 32, 64, or 128 examples) and compute the gradient — and perform one update — per batch:

```
∂L/∂w ≈ (1/b) · Σᵢ∈batch ∂Lᵢ/∂w        (b = batch size)
```

This is the **default choice in essentially all modern deep learning**, for a simple reason: it captures most of batch gradient descent's computational efficiency (vectorized matrix operations across the batch) while still updating frequently enough to converge quickly, without needing the entire dataset in memory at once.

---

## 5. Comparing the Three Paths

Running all three methods on the same regression problem (fitting `y = w₁x₁ + w₂x₂` to noisy data) from the same starting point makes the difference in trajectory shape immediately visible:

![Batch, mini-batch, and SGD trajectories on the same loss surface](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/gd_trajectory_comparison.png)

*Batch gradient descent (left) traces a smooth, direct curve toward the minimum — but only gets 15 such steps total (one per epoch) in this example. Mini-batch (middle) is mostly smooth with some jitter from batch-to-batch variation. SGD (right) is visibly noisy, but note it's also had **far more updates** in the same number of epochs — that difference in update count is exactly what the next section explains.*

The difference in update frequency is stark for any realistically-sized dataset:

![Update frequency comparison](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/update_frequency.png)

*For a 1,000-row dataset with a mini-batch size of 32: batch gradient descent updates once per epoch, mini-batch updates 31 times, and SGD updates all 1,000 times — a three-order-of-magnitude difference in how often the parameters actually move.*

---

## 6. Convergence Speed in Practice

Because SGD and mini-batch update so much more often, they typically reach a good loss value faster in terms of **wall-clock time or epochs elapsed**, even though each individual update is noisier and less precise than batch gradient descent's:

![Loss curves: smooth batch descent vs. noisy but faster SGD and mini-batch](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/loss_curves_comparison.png)

*Plotted against the same epoch axis, SGD and mini-batch both reach a low loss within roughly 1–2 epochs' worth of updates, while batch gradient descent — despite its smooth, "textbook" trajectory — is still slowly working its way down after 15 full epochs. This is the practical reason SGD and mini-batch dominate in deep learning: more frequent, noisier updates usually beat fewer, more "correct" ones, simply because there are so many more of them.*

---

## 7. A Hidden Benefit of Noise: Escaping Local Minima

Real loss landscapes (especially in deep networks) are rarely a single smooth bowl — they can have multiple local minima. Batch gradient descent's gradient is exact but also *fully deterministic*: once it settles into a local minimum, there's nothing to push it back out.

![How SGD's noise can help escape a shallow local minimum](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/escaping_local_minima.png)

*Both paths start in the basin of a shallower, worse local minimum. Batch gradient descent (blue) follows the exact gradient straight down into it and stays there permanently. SGD's per-example noise (red) means its steps don't always point in the exact downhill direction — occasionally, that noise is enough to carry it back over the barrier into a deeper, better minimum on the other side.*

This is a real, well-documented phenomenon (not guaranteed on every run — it depends on the noise happening to point the right way at the right time), and it's one of the reasons SGD-style noise is sometimes deliberately preserved even when more updates could be batched together.

---

## 8. Implementation Tip: Why Batch Sizes Are Powers of 2

Two practical questions come up immediately when actually setting a batch size:

![Batch sizes as powers of 2, and handling the remainder batch](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/batch_size_practical.png)

- **Why powers of 2 (32, 64, 128, 256...)?** GPUs and TPUs allocate and move memory in fixed-size chunks tied to their hardware architecture. A batch size that's a power of 2 tiles those chunks exactly, with no wasted or misaligned memory; an arbitrary size like 30 can leave partially-used chunks, which slightly hurts throughput.
- **What happens if the batch size doesn't evenly divide the dataset?** For example, 1,000 rows with a batch size of 32 gives 31 full batches (992 rows) plus a smaller **remainder batch of 8**. This last, undersized batch is simply used as-is — every major framework (Keras, PyTorch, etc.) handles this automatically without any special configuration needed.

---

## 9. Summary Table

| | Batch GD | Mini-Batch GD | SGD |
|---|---|---|---|
| Gradient computed from | Entire dataset | Small batch (e.g. 32) | 1 example |
| Updates per epoch | 1 | dataset size / batch size | dataset size |
| Path to minimum | Smooth | Mostly smooth, some noise | Noisy, erratic |
| Vectorization | Excellent | Good | None (one row at a time) |
| Memory required | Whole dataset at once | One small batch at a time | One row at a time |
| Can escape shallow local minima | No | Sometimes | Yes (most often) |
| Typical use today | Rare (small datasets only) | **Default choice** | Rare in pure form; conceptually important |

---

## 10. Code: All Three Variants

```python
import numpy as np

def train(X, y, lr=0.1, epochs=20, batch_size=None):
    """
    batch_size=None      -> Batch Gradient Descent (uses the full dataset each update)
    batch_size=1         -> Stochastic Gradient Descent
    batch_size=32 (etc.) -> Mini-Batch Gradient Descent
    """
    n, n_features = X.shape
    w = np.zeros(n_features)
    losses = []

    effective_batch = n if batch_size is None else batch_size

    for epoch in range(epochs):
        perm = np.random.permutation(n)          # shuffle each epoch
        X_shuffled, y_shuffled = X[perm], y[perm]

        for start in range(0, n, effective_batch):
            end = start + effective_batch          # naturally handles the remainder batch
            Xb, yb = X_shuffled[start:end], y_shuffled[start:end]

            pred = Xb @ w
            error = yb - pred
            grad = -2 * Xb.T @ error / len(Xb)      # averaged over the current (mini-)batch
            w = w - lr * grad

        losses.append(np.mean((y - X @ w) ** 2))
    return w, losses

# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------
rng = np.random.default_rng(42)
X = rng.normal(0, 1, (1000, 2))
true_w = np.array([3.0, 2.0])
y = X @ true_w + rng.normal(0, 0.5, 1000)

w_batch, losses_batch = train(X, y, lr=0.1, epochs=20, batch_size=None)  # batch GD
w_sgd, losses_sgd = train(X, y, lr=0.01, epochs=5, batch_size=1)         # SGD
w_mb, losses_mb = train(X, y, lr=0.05, epochs=10, batch_size=32)         # mini-batch

print("Batch GD final weights:", w_batch)
print("SGD final weights:     ", w_sgd)
print("Mini-batch final weights:", w_mb)
```

**Using Keras**, the batch size is simply a parameter to `.fit()` — the three variants aren't separate APIs, just different values of one argument:

```python
model.fit(X, y, batch_size=len(X), epochs=20)   # batch GD (batch size = full dataset)
model.fit(X, y, batch_size=1, epochs=5)         # SGD
model.fit(X, y, batch_size=32, epochs=10)       # mini-batch (the default in practice)
```

---

## 11. Key Takeaways

- All three variants use the exact same update rule, `w ← w − η·∂L/∂w` — they differ only in how much data goes into computing that gradient before each update.
- **Batch GD**: one precise, stable update per epoch, but slow to make any progress and expensive on large datasets.
- **SGD**: one (noisy) update per training example — far more frequent updates, often converging faster in practice despite the noise, and occasionally able to escape shallow local minima that would trap batch GD permanently.
- **Mini-batch GD**: the practical default, balancing vectorized efficiency against update frequency and memory footprint.
- Batch sizes are conventionally chosen as powers of 2 to align cleanly with GPU/TPU memory architecture; a final undersized "remainder" batch is handled automatically by standard frameworks.

---

## 12. Further Reading

- Robbins, H. & Monro, S. (1951). *A Stochastic Approximation Method* — the foundational paper behind stochastic optimization.
- Bottou, L. (2010). *Large-Scale Machine Learning with Stochastic Gradient Descent* — a thorough treatment of SGD's practical behavior and trade-offs.
- Keskar, N. et al. (2016). *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima* — on how batch size choice affects which kind of minimum training converges to.
- See also this repo's [`backprop-part1.md`](backprop-part1.md) and [`backprop-part2.md`](backprop-part2.md) for how the gradient being averaged here is actually computed via the chain rule.

---

*Diagrams in this document were generated programmatically (including running actual gradient descent simulations to produce the trajectory and loss-curve figures) and are hosted in this repo's [`images/`](https://github.com/versatilerajan/deepcontent/tree/main/images) folder.*
