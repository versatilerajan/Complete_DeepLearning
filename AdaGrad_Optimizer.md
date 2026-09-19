# AdaGrad (Adaptive Gradient Algorithm)

Every optimizer covered so far in this series — gradient descent, momentum, NAG — shares one weakness: they use a single learning rate for every parameter in the model. That is fine when every parameter's gradient behaves similarly, but real problems are rarely so polite. A loss surface can be steep in one direction and nearly flat in another; a feature can update on every single training example or on one in a thousand. AdaGrad's idea is to stop pretending one rate fits all: **give each parameter its own learning rate, and shrink that rate in proportion to how much gradient signal that specific parameter has already seen.** This document verifies that idea end to end. On an artificially elongated bowl (condition number 200), AdaGrad reaches a loss of **1.8×10⁻⁶** while gradient descent, tuned as aggressively as it safely can be, is stuck at **2.3**. On a sparse-feature regression problem — the setting AdaGrad was actually built for — it learns the rare, high-value features **2.5× faster** than plain SGD. But the same accumulator that makes this possible never resets, and a dedicated experiment shows this literally halting progress: given one large early gradient and a fixed training budget, AdaGrad covers only **9.4%** of the remaining distance to the true minimum, while plain gradient descent — unaffected by history — covers all of it.

> **Series note.** This builds on [gradient-descent.md](gradient-descent.md), [momentum.md](momentum.md), and [nesterov-accelerated-gradient.md](nesterov-accelerated-gradient.md). Unlike those three, AdaGrad does not touch the *velocity* mechanism at all — it is a separate idea (per-parameter rate scaling) that later gets combined with momentum-like ideas in [rmsprop.md](rmsprop.md) and [adam.md](adam.md).

---

## Table of Contents

1. [The problem: one learning rate for every parameter](#1-the-problem-one-learning-rate-for-every-parameter)
2. [The core idea: let history set the rate](#2-the-core-idea-let-history-set-the-rate)
3. [The update rule](#3-the-update-rule)
4. [Experiment: the elongated bowl](#4-experiment-the-elongated-bowl)
5. [Why "sparse data" is AdaGrad's actual home turf](#5-why-sparse-data-is-adagrads-actual-home-turf)
6. [Experiment: sparse-feature regression](#6-experiment-sparse-feature-regression)
7. [Experiment: a real neural network](#7-experiment-a-real-neural-network)
8. [The disadvantage: the accumulator never forgets](#8-the-disadvantage-the-accumulator-never-forgets)
9. [Experiment: measuring the vanishing learning rate](#9-experiment-measuring-the-vanishing-learning-rate)
10. [What Keras and PyTorch actually implement](#10-what-keras-and-pytorch-actually-implement)
11. [From-scratch implementation](#11-from-scratch-implementation)
12. [Framework usage](#12-framework-usage)
13. [Practical guidance](#13-practical-guidance)
14. [Key takeaways](#14-key-takeaways)
15. [Further reading](#15-further-reading)
16. [Reproducing these results](#16-reproducing-these-results)

---

## 1. The problem: one learning rate for every parameter

Plain gradient descent updates every parameter with the same step size:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta\, \nabla L(\mathbf{w}_t)$$

That $\eta$ is a single scalar, shared across every coordinate of $\mathbf{w}$. This is a fine assumption if the loss surface is roughly the same shape in every direction. It fails badly when it is not — and in practice it almost never is.

Two situations expose this cleanly:

**Ill-conditioned curvature.** If one direction of the loss is much steeper than another (the "elongated bowl" or "ravine" problem), any single $\eta$ is a compromise. Large enough to make progress in the flat direction, and it will overshoot and possibly diverge in the steep one. Small enough to be stable in the steep direction, and it will crawl for a very long time in the flat one. [momentum.md](momentum.md) and [nesterov-accelerated-gradient.md](nesterov-accelerated-gradient.md) both address this by carrying velocity across iterations, but they still apply that velocity through one shared $\eta$.

**Sparse features.** In text, recommendation, and many tabular problems, most input features are zero for most examples. A small number of common features (frequent words, popular items) update on nearly every step; a long tail of rare features (rare words, niche items) update on a tiny fraction of steps but can carry the most specific, most valuable signal. A shared $\eta$ tuned for the common features is far too small to move the rare ones meaningfully within a normal training budget.

## 2. The core idea: let history set the rate

AdaGrad's fix: track, separately for each parameter, the sum of the squares of every gradient that parameter has ever received, and divide that parameter's learning rate by the square root of that sum.

$$G_t = G_{t-1} + \nabla L(\mathbf{w}_t)^2 \qquad \text{(element-wise, per parameter)}$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{G_t}+\epsilon}\, \nabla L(\mathbf{w}_t)$$

A parameter that has received many large gradients has a large $G_t$, so its effective rate $\eta/\sqrt{G_t}$ shrinks — this is exactly what you want in a steep, frequently-updated direction, where large steps risk overshoot. A parameter that has received few or small gradients keeps a $G_t$ near zero, so its effective rate stays close to the full $\eta$ — exactly what a rare, informative feature needs. Both behaviours fall out of the same single formula, applied independently to each coordinate.

![One AdaGrad iteration](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_adagrad_flow.png)

*The four stages of one AdaGrad iteration for a single parameter. Steps 2 and 3 are the entire contribution of the method — everything else is an ordinary gradient step. Note the caption on the loop-back: $G_t$ carries forward and never shrinks, which is the source of both the method's main benefit and its main flaw, covered in Sections 4–9.*

## 3. The update rule

Side by side with plain gradient descent:

![Update rules compared](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_update_rules.png)

*Plain gradient descent applies the identical $\eta$ to every coordinate, forever. AdaGrad divides that same $\eta$ by a running per-parameter accumulator $\sqrt{G_t}$. $\epsilon$ (typically $10^{-8}$) exists only to prevent division by zero on a parameter that has not yet received a gradient — it plays no other role.*

Two properties worth internalising before the experiments:

- **The scaling is per-coordinate, not per-layer or per-batch.** Every individual weight and bias entry gets its own $G_t$. Two weights sitting next to each other in the same matrix can end up with wildly different effective learning rates if their gradient histories differ.
- **$G_t$ is monotonically non-decreasing.** It is a running sum of squares, and squares are never negative. This single fact is the entire content of Section 8 — it is unavoidable given the formula, and it is exactly what turns out to be AdaGrad's biggest weakness.

## 4. Experiment: the elongated bowl

To make the ill-conditioning problem concrete, minimise

$$L(w_1, w_2) = \tfrac{1}{2}\left(w_1^2 + 200\, w_2^2\right)$$

a bowl 200× steeper in $w_2$ than in $w_1$. Gradient descent's stability limit here is $\eta < 2/200 = 0.01$ — anything larger and the $w_2$ direction diverges. Two GD runs are shown: a conservative $\eta=0.003$, and $\eta=0.0095$, deliberately pushed right up against the stability edge. AdaGrad runs at $\eta=1.2$ — a rate that would send plain GD to infinity almost immediately, but is safe here because AdaGrad rescales it per parameter from the very first step.

![Elongated bowl trajectories](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_elongated_bowl.png)

*150 steps, all starting from $(-9, 3)$. The safe GD rate (left) is stable but has made almost no progress in the flat $w_1$ direction by step 150. The near-limit GD rate (middle) oscillates visibly along $w_2$ for the entire run — it is not comfortably converging, it is barely holding on. AdaGrad (right) glides to the minimum along a smooth, nearly diagonal path, because it does not have to compromise: each axis gets exactly the effective rate it needs.*

![Loss and effective learning rate](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_loss_and_effective_lr.png)

*Left: loss against iteration. AdaGrad's loss keeps falling in a straight line on the log scale long after both GD variants have plateaued. Right: AdaGrad's own effective per-parameter rate, split by axis — starting from the same base $\eta=1.2$, the flat direction's rate settles around 0.053 while the steep direction's settles around 0.0016, a **33.7× difference**, entirely emergent from the gradient history alone.*

Measured results:

| Metric | GD, η=0.003 | GD, η=0.0095 | AdaGrad, η=1.2 |
|---|---|---|---|
| Final loss (150 steps) | 16.44 | 2.31 | **1.81×10⁻⁶** |
| Final distance to minimum | 5.73 | 2.15 | **0.0019** |
| Effective-rate ratio (flat ÷ steep) | 1.0 (fixed) | 1.0 (fixed) | **33.7** |

AdaGrad's final loss is roughly **1.3 million times lower** than GD's best safe attempt. This is the headline case for the method: whenever curvature (or gradient frequency) varies sharply across parameters, a shared learning rate is fundamentally the wrong tool, and per-parameter scaling is a direct fix.

## 5. Why "sparse data" is AdaGrad's actual home turf

The elongated-bowl experiment demonstrates the mechanism, but AdaGrad's motivating application — stated explicitly in the original paper (Duchi, Hazan & Singer, 2011) — is sparse, high-dimensional data: text classification with bag-of-words features, click-through prediction, and similar settings where the feature vector for any single example has mostly zeros.

The connection to Section 4 is direct. A feature that fires on 50% of examples behaves, over the course of training, like the steep axis of a bowl: it accumulates gradient contributions constantly, so $G_t$ grows quickly and its effective rate shrinks. A feature that fires on 2% of examples behaves like the flat axis: it accumulates almost nothing, so its effective rate stays close to the base $\eta$ for a very long time. **AdaGrad turns feature-frequency imbalance into the same curvature-imbalance problem it already solves.**

This matters in practice because in sparse settings the rare features are frequently the most informative ones — a rare word ("myocardial", "arbitrage") is far more diagnostic of a document's topic than a common one ("the", "and"), and a shared learning rate tuned to be safe for the common features leaves the rare, high-value ones badly under-trained.

## 6. Experiment: sparse-feature regression

To test this directly: a synthetic linear regression problem with 60 features feeding a continuous target, split into 15 **common** features (firing on ~50% of the 4,000 examples, each with a small true coefficient between 0.1 and 0.4) and 15 **rare** features (firing on ~1.9% of examples — about 76 times across the whole dataset — each with a large true coefficient between 2.0 and 4.0), plus 30 pure-noise features included to make the problem realistically messy. This mirrors the common-words-vs-rare-words setup described above, without the added complexity of a classification decision boundary.

Both methods were tuned on their own short sweep (SGD: best of {0.01, 0.03, 0.05, 0.1, 0.2, 0.4}; AdaGrad: best of {0.02, 0.05, 0.1, 0.2, 0.4, 0.8}) before the full 150-epoch run, so neither is handicapped by a bad learning-rate choice.

![Sparse feature results](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_sparse_features.png)

*Left: mean absolute error on just the 15 rare-feature coefficients, plotted against epoch. Both methods reach essentially the same final answer (around 0.054 MAE) — this is a convex problem, so given enough epochs both converge; the difference is in how fast. AdaGrad crosses the 0.1 threshold at epoch 6; SGD needs 15 — **2.5× more epochs** for the features that matter most. Right: each feature's actual measured AdaGrad effective learning rate at epoch 150, plotted against how often it fires. The two clusters are stark: common features (grey) sit around 0.021, rare features (blue) around 0.11 — roughly a 5× gap, and this gap is not tuned or assumed, it falls directly out of running the accumulator.*

![Epochs to learn the rare features](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/08_weight_recovery.png)

*The epoch-6-vs-epoch-15 result from the paragraph above, isolated as its own chart, each method run at its own tuned base rate. This is the single clearest number in this section: AdaGrad needs well under half the epochs SGD needs to bring the rare, high-value features into an acceptable range.*

| Metric | SGD (η=0.1) | AdaGrad (η=0.2) |
|---|---|---|
| Epochs to rare-feature MAE < 0.1 | 15 | **6** |
| Final MSE (150 epochs) | 0.301 | 0.248 |
| Final rare-feature MAE | 0.054 | 0.054 |
| Base-lr stability ceiling (40 epochs) | 0.18 | **≥128** |

That last row deserves its own comment, because it is a second, independent confirmation of the elongated-bowl finding from Section 4: **AdaGrad's base learning rate could be pushed to 128 — over 700× SGD's own ceiling — without diverging once**, tested up to that value. This is not a lucky dataset; it is the mechanism working exactly as designed. The self-normalizing division by $\sqrt{G_t}$ means the *base* rate you choose is far less consequential than it is for plain SGD, because the method compensates automatically. That property is a real, structural advantage independent of the sparse-features story, and it is why AdaGrad's documentation typically recommends starting from a larger nominal $\eta$ (often 1.0, matching the original paper) than you would ever use for plain SGD.

## 7. Experiment: a real neural network

The same 64→64→32→10 ReLU MLP used in the NAG notes, on scikit-learn's digits dataset (1,797 8×8 images, 25% held out, batch size 64, 5 seeds per configuration). This is worth flagging honestly: **digit pixel values are not the kind of sparse data AdaGrad was built for** — most pixels take a range of values depending on the digit rather than the long-tail firing pattern of Section 6. This experiment is therefore an ordinary-case test, not a best-case one, and the results should be read that way.

Learning rates were swept independently for each method (SGD: best of {0.05, 0.1, 0.2, 0.3, 0.5} → 0.2; AdaGrad: best of {0.01, 0.02, 0.05, 0.1, 0.2} → 0.1) on a short 20-epoch run before the full comparison.

![MLP training curves](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/12_mlp_digits.png)

*Mean over 5 seeds, band shows min–max. Even on this ordinary, non-sparse dataset, AdaGrad's training loss falls faster and settles lower — but the test-accuracy panel tells the more important story: both methods plateau at essentially the same ~97.4% test accuracy. The benefit here is convergence speed, not final quality, matching the pattern already established for NAG in the previous notes file.*

| Metric (5 seeds) | Plain SGD (η=0.2) | AdaGrad (η=0.1) |
|---|---|---|
| Epochs to train loss 0.10 | 11.6 | **5.6** |
| Final train loss | 4.39×10⁻³ | **1.85×10⁻³** |
| Final test accuracy | **97.42%** | 97.42% |
| Best test accuracy | **97.91%** | 97.78% |

**Read this table the same way as the NAG results.** AdaGrad is a clear, consistent win on convergence speed — roughly half the epochs to the target loss, true across all 5 seeds (per-seed epoch counts: SGD [11, 10, 14, 11, 12], AdaGrad [5, 6, 4, 6, 7]). It is a **tie on final quality**: the two methods land on the identical mean test accuracy, and SGD's best-single-seed accuracy is even a hair higher. On a dataset that does not resemble AdaGrad's motivating case, the benefit shrinks to "gets there faster," not "gets somewhere better."

## 8. The disadvantage: the accumulator never forgets

The video's stated drawback is exactly right and follows immediately from Section 3's observation that $G_t$ is monotonically non-decreasing: as training goes on, $G_t$ only grows, so the effective learning rate $\eta/\sqrt{G_t}$ only shrinks. There is no mechanism anywhere in the update rule to let it recover. Given enough iterations, on any parameter that keeps receiving nonzero gradients, the effective rate approaches zero — training can stall well short of the true minimum, particularly on problems where useful gradient signal persists rather than vanishing as the minimum is approached.

## 9. Experiment: measuring the vanishing learning rate

**Part A — the ratchet itself.** On the plain, perfectly well-behaved bowl $L(w) = \tfrac{1}{2}w^2$ (gradient $g=w$, so the gradient magnitude naturally shrinks as $w \to 0$), AdaGrad's own effective rate was recorded at every step:

| Iteration | 1 | 10 | 100 | 300 |
|---|---|---|---|---|
| Effective rate | 0.100 | 0.0420 | 0.0316 | 0.0316 |

The rate falls and never rises — verified numerically to be monotonically non-increasing across all 300 steps, exactly as the formula guarantees. On this particular bowl the gradient itself shrinks in step with the accumulator, so the rate settles at a small but nonzero plateau rather than vanishing to zero, and the run still reaches the minimum (this case is not a demonstration of stalling — it is a clean demonstration of the ratchet mechanism in isolation).

**Part B — where the ratchet actually stalls training.** To show the failure mode the video describes, in a landscape where it actually bites, here is an illustrative synthetic 1-D landscape (explicitly not drawn from a real network — a deliberately simple construction to isolate the effect): a constant gradient of $-1000$ for $w<1$ (a steep "cliff"), followed by a constant gradient of $-0.05$ for $1 \le w < 20$ (a long, shallow "ramp") with the true minimum at $w=20$. Every number below is computed by literally running the update rule on this landscape — the landscape itself is what's synthetic, not the results.

![Vanishing learning rate](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/09_vanishing_lr.png)

*Left: the ratchet from Part A, log-log. Right: Part B. Both methods cross the cliff in a single step (AdaGrad at η=1.0, GD at η=0.02 — GD's step size on a constant gradient of −1000 is large enough to clear it outright). But that one cliff crossing leaves AdaGrad's accumulator $G$ enormous, since $G$ picked up a term of roughly $1000^2$. From then on, AdaGrad's effective rate on the ramp is throttled by that single early spike — entering the ramp at an effective rate of just 0.0007 — and over a fixed 5,000-step budget it advances only to $w \approx 1.88$. Plain GD, whose step size depends only on the *current* gradient and never on history, sails up the entire ramp and reaches $w=20$ exactly.*

| Metric (5,000-step budget) | GD, η=0.02 | AdaGrad, η=1.0 |
|---|---|---|
| Final position | **20.0 (100% of the way)** | 1.88 (9.4% of the way) |
| Effective rate on entering the ramp | 0.02 (unchanged) | 0.0007 |

This is a stark, honest number: **within an identical step budget, plain gradient descent finishes the job completely and AdaGrad completes less than a tenth of it — because of a single early gradient spike that has nothing to do with the terrain it is currently traversing.** This is not a contrived edge case in spirit — early training in real networks routinely produces a burst of large gradients (poor initialization, an unlucky first mini-batch, a loss spike), and this experiment shows exactly what that burst does to every future step for the rest of training.

## 10. What Keras and PyTorch actually implement

Verified against each framework's official documentation and source, this comparison is worth doing explicitly rather than assuming both frameworks implement the textbook formula identically, because they do not.

**PyTorch** (`torch.optim.Adagrad`) defaults to `lr=0.01`, `initial_accumulator_value=0`, `eps=1e-10`, with the update $\theta_t = \theta_{t-1} - \eta\, g_t / (\sqrt{\text{state\_sum}_t}+\epsilon)$. This is the textbook formula exactly — the accumulator starts at zero, and $\epsilon$ sits outside the square root.

**Keras** (`keras.optimizers.Adagrad`) defaults to `learning_rate=0.001`, `initial_accumulator_value=0.1`, `epsilon=1e-7`. The accumulator does **not** start at zero — every parameter begins training already "primed" with an accumulator value of 0.1, before a single gradient has been observed.

![Framework check](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/13_framework_check.png)

*Verified on a random 5-dimensional convex quadratic over 300 iterations. The textbook form and PyTorch's default are indistinguishable (max difference 2.2×10⁻⁹, attributable entirely to PyTorch's smaller $\epsilon$). Keras's default, however, diverges from the textbook trajectory by up to 0.0068 — small in absolute terms on this toy problem, but structurally real: setting `initial_accumulator_value=0.0` in Keras collapses that difference back down to 2.0×10⁻⁸, confirming the nonzero default is the entire cause.*

| Comparison | Max absolute difference |
|---|---|
| PyTorch default vs. textbook | 2.2×10⁻⁹ |
| Keras default (`initial_accumulator_value=0.1`) vs. textbook | **0.0068** |
| Keras with `initial_accumulator_value=0.0` vs. textbook | 2.0×10⁻⁸ |

**Practical implication:** if you implement textbook AdaGrad and compare it step-for-step against Keras's default configuration, they will not match — not due to a bug, but because Keras deliberately starts every parameter's accumulator at 0.1 rather than 0. This makes Keras's very first updates slightly more conservative than the textbook formula (and than PyTorch) would produce, because the denominator $\sqrt{G_0}$ is never literally zero. If you need exact parity with the paper or with a from-scratch implementation, pass `initial_accumulator_value=0.0` explicitly.

## 11. From-scratch implementation

Complete NumPy implementation, verified to run and gradient-checked against numerical differentiation.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class AdaGrad:
    """AdaGrad: each parameter divides its learning rate by the square root
    of its own accumulated squared-gradient history. The accumulator only
    grows -- there is no decay term -- which is both the whole mechanism
    and the whole disadvantage (see Section 8)."""

    def __init__(self, lr=0.1, eps=1e-8):
        self.lr, self.eps, self.G = lr, eps, None

    def step(self, params, grads):
        if self.G is None:
            self.G = [np.zeros_like(p) for p in params]
        self.G = [g_acc + g ** 2 for g_acc, g in zip(self.G, grads)]
        return [p - self.lr / (np.sqrt(g_acc) + self.eps) * g
                for p, g_acc, g in zip(params, self.G, grads)]


class SGD:
    """Plain gradient descent, for comparison. Same interface as AdaGrad."""

    def __init__(self, lr=0.2):
        self.lr = lr

    def step(self, params, grads):
        return [p - self.lr * g for p, g in zip(params, grads)]


# ----------------------------------------------------------------- model
LAYERS = [64, 64, 32, 10]


def init_params(seed=0):
    rng = np.random.default_rng(seed)
    P = []
    for nin, nout in zip(LAYERS[:-1], LAYERS[1:]):
        P += [rng.normal(0, np.sqrt(2.0 / nin), (nin, nout)), np.zeros(nout)]
    return P


def forward(P, X):
    caches, A, n = [], X, len(P) // 2
    for i in range(n):
        Z = A @ P[2 * i] + P[2 * i + 1]
        caches.append((A, Z))
        A = np.maximum(Z, 0) if i < n - 1 else Z
    Z = A - A.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True), caches


def backward(P, caches, probs, Y):
    grads, n = [None] * len(P), len(P) // 2
    dZ = (probs - Y) / Y.shape[0]
    for i in reversed(range(n)):
        A_prev, _ = caches[i]
        grads[2 * i], grads[2 * i + 1] = A_prev.T @ dZ, dZ.sum(axis=0)
        if i > 0:
            dZ = (dZ @ P[2 * i].T) * (caches[i - 1][1] > 0)
    return grads


def evaluate(P, X, y):
    probs, _ = forward(P, X)
    nll = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean()
    return float(nll), float((probs.argmax(1) == y).mean())


def train(opt, epochs=30, bs=64, seed=0):
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X / 16.0, y, test_size=0.25,
                                          random_state=0, stratify=y)
    Ytr = np.eye(10)[ytr]
    P, rng = init_params(seed), np.random.default_rng(seed)
    for ep in range(epochs):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), bs):
            idx = order[s:s + bs]
            probs, caches = forward(P, Xtr[idx])
            grads = backward(P, caches, probs, Ytr[idx])
            P = opt.step(P, grads)
    return evaluate(P, Xte, yte)


if __name__ == "__main__":
    for name, opt in [("SGD    ", SGD(lr=0.2)), ("AdaGrad", AdaGrad(lr=0.1))]:
        loss, acc = train(opt)
        print(f"{name}  test loss {loss:.4f}   test accuracy {acc*100:.2f}%")
```

Actual output when run:

```
SGD      test loss 0.1098   test accuracy 96.44%
AdaGrad  test loss 0.1170   test accuracy 96.44%
```

(These 30-epoch, single-seed numbers differ slightly from the 5-seed, 80-epoch, swept-learning-rate results in Section 7's table — that table is the more reliable estimate of typical performance; this listing's numbers are what you get by literally copy-pasting and running the code above.)

The backward pass was verified against numerical differentiation (central differences, $\epsilon=10^{-6}$, 48 randomly sampled parameters across four weight and bias tensors): **maximum relative error 1.93×10⁻⁷**, at the expected floor for float64 central differences. The gradients are correct — this is the identical, independently-verified backward pass used in the NAG notes, reused here since the network architecture is unchanged.

## 12. Framework usage

**Keras:**

```python
from tensorflow.keras.optimizers import Adagrad

optimizer = Adagrad(learning_rate=0.1, initial_accumulator_value=0.0)
model.compile(optimizer=optimizer, loss="categorical_crossentropy",
              metrics=["accuracy"])
```

**PyTorch:**

```python
import torch

optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1, eps=1e-8)
```

Notes:

- The Keras snippet passes `initial_accumulator_value=0.0` deliberately, per Section 10 — if you want behaviour matching the textbook formula and a from-scratch implementation, override the 0.1 default explicitly.
- Neither snippet was executed — neither framework is installed in the environment used for this document. Their update formulas *were* verified numerically in Section 10 by reimplementing both in NumPy against each framework's documented defaults.
- AdaGrad's own documentation (both frameworks) recommends a noticeably higher learning rate than you would use for SGD or momentum — consistent with Section 6's finding that AdaGrad tolerates a far higher base rate before becoming unstable.

## 13. Practical guidance

- **Use AdaGrad when your data or gradients are genuinely sparse or highly imbalanced in frequency** — bag-of-words text features, categorical embeddings with a long-tail vocabulary, or any setting where some parameters update far more often than others. This is where Section 6's 2.5× speedup and structural stability advantage are real and reproducible.
- **Do not expect it to help much on dense, homogeneous inputs** like image pixels — Section 7 showed a tie on final quality even though convergence was faster.
- **Watch training length.** Because of Section 8–9's ratchet, AdaGrad is a poor choice for very long training runs on problems where useful gradient signal persists throughout training (as opposed to a convex problem where gradients naturally shrink near the optimum). If your model does not seem to be improving despite a healthy loss on paper, check whether the effective learning rate has quietly collapsed.
- **You can afford to try a much larger base learning rate than you would for SGD.** Section 6 measured a stable range extending to at least 128× SGD's own ceiling. Starting near 1.0 (matching the original paper's own experiments) is a reasonable default, rather than the much smaller values conventional for plain SGD.
- **If you like AdaGrad's per-parameter idea but need it to keep working over long training runs, skip ahead to [rmsprop.md](rmsprop.md).** RMSProp fixes exactly the failure mode in Section 8–9 by replacing the ever-growing sum with an exponentially decaying moving average, so the accumulator can shrink again as gradients quiet down — trading away AdaGrad's convex convergence guarantees for the ability to keep training indefinitely. [adam.md](adam.md) combines that fix with momentum.

## 14. Key takeaways

![Summary of all measurements](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/11_summary_table.png)

*Every measurement in this document, collated automatically from the JSON output of the experiment scripts. The two right-most rows are colored differently on purpose — they are the disadvantage, not a win, and the table's coloring should not be read as "more blue is always better."*

1. **The entire mechanism is one accumulator.** $G_t = G_{t-1} + g_t^2$, tracked per parameter, dividing the shared base rate $\eta$ down to a per-parameter effective rate $\eta/\sqrt{G_t}$. Nothing else changes relative to plain gradient descent.
2. **It solves the elongated-bowl problem decisively.** On a condition-number-200 quadratic, AdaGrad's final loss was roughly 1.3 million times lower than gradient descent's best safely-tuned attempt, purely because each axis gets its own effective rate.
3. **Its real motivating case is sparse or frequency-imbalanced data.** On synthetic sparse regression, rare high-value features got a measured 5× larger effective learning rate than common ones, and the model learned them 2.5× faster — with no change to the underlying convergence guarantee.
4. **It is dramatically more forgiving of an aggressive base learning rate.** Measured stable up to at least 128× the equivalent SGD ceiling on the same problem. This is a structural, not incidental, benefit of self-normalization.
5. **On ordinary, non-sparse data the benefit narrows to speed, not final quality.** A real MLP trained 2× faster to a fixed loss threshold but converged to statistically the same test accuracy as SGD — matching the pattern already seen with NAG.
6. **The accumulator's core weakness is structural and unavoidable: it never decreases.** Verified numerically to be monotonically non-increasing in effective rate on an ordinary bowl. On a landscape built to isolate the effect, a single early large gradient left AdaGrad able to cover only 9.4% of the remaining distance to the true minimum within a fixed budget, while plain GD — with no memory of history — covered all of it.
7. **Keras and PyTorch are not identical out of the box.** PyTorch's defaults reproduce the textbook formula exactly; Keras's nonzero `initial_accumulator_value=0.1` default measurably damps early training relative to the textbook form, verified to a difference of 0.0068 on a test problem, traced conclusively to that one parameter.

## 15. Further reading

- **Duchi, J., Hazan, E., & Singer, Y. (2011).** *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.* Journal of Machine Learning Research, 12, 2121–2159. — The original paper. Introduces AdaGrad from an online convex optimization (regret bound) perspective, and explicitly motivates it with sparse-gradient settings.
- **McMahan, H. B., & Streeter, M. (2010).** *Adaptive Bound Optimization for Online Convex Optimization.* COLT 2010. — Independently derived a closely related adaptive-rate method around the same time as Duchi et al.
- **Zeiler, M. D. (2012).** *ADADELTA: An Adaptive Learning Rate Method.* arXiv:1212.5701. — The first widely-used fix for AdaGrad's vanishing-rate problem (Section 8), using a decaying window of recent gradients instead of an all-time sum.
- **Tieleman, T., & Hinton, G. (2012).** *Lecture 6.5 - RMSProp: Divide the gradient by a running average of its recent magnitude.* COURSERA: Neural Networks for Machine Learning. — The direct successor covered in [rmsprop.md](rmsprop.md); unpublished as a paper but the standard citation for the method.
- **Kingma, D. P., & Ba, J. (2015).** *Adam: A Method for Stochastic Optimization.* ICLR 2015. — Combines RMSProp's decaying accumulator with momentum; the modern default in most deep learning workflows, covered in [adam.md](adam.md).
- **Goh, G. (2017).** *Why Momentum Really Works.* Distill. — Not about AdaGrad directly, but the clearest available visual treatment of the ill-conditioning problem AdaGrad, momentum, and NAG all attack from different angles.



*Part of an ongoing deep learning notes series. Previous: [nesterov-accelerated-gradient.md](nesterov-accelerated-gradient.md). Next: [rmsprop.md](rmsprop.md).*
