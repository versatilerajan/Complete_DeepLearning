# RMSprop: Fixing AdaGrad's Vanishing Learning Rate

*Companion notes on RMSprop — how it builds directly on AdaGrad's idea of a per-parameter learning rate, the exact flaw in AdaGrad that causes training to stall prematurely, and how swapping a running sum for an EWMA (from the [EWMA notes](ewma.md)) fixes it — every claim backed by a real trained-network experiment.*

---

## Table of Contents

1. [Recap: Why Plain Gradient Descent Needed Fixing](#1-recap-why-plain-gradient-descent-needed-fixing)
2. [AdaGrad: Giving Every Parameter Its Own Learning Rate](#2-adagrad-giving-every-parameter-its-own-learning-rate)
3. [AdaGrad's Fatal Flaw: The Vanishing Learning Rate](#3-adagrads-fatal-flaw-the-vanishing-learning-rate)
4. [Real Experiment: Watching the Accumulator Grow](#4-real-experiment-watching-the-accumulator-grow)
5. [RMSprop's Fix: an EWMA Instead of a Sum](#5-rmsprops-fix-an-ewma-instead-of-a-sum)
6. [Real Experiment: Does the Effective Learning Rate Actually Stay Healthy?](#6-real-experiment-does-the-effective-learning-rate-actually-stay-healthy)
7. [Real Experiment: A Full Training Comparison](#7-real-experiment-a-full-training-comparison)
8. [Summary Table](#8-summary-table)
9. [Practical Implementation in Keras](#9-practical-implementation-in-keras)
10. [Code: AdaGrad and RMSprop From Scratch](#10-code-adagrad-and-rmsprop-from-scratch)
11. [Key Takeaways](#11-key-takeaways)
12. [Further Reading](#12-further-reading)

---

## 1. Recap: Why Plain Gradient Descent Needed Fixing

The [optimizers introduction](optimizers-part1.md) identified a specific weakness in plain gradient descent: it applies the **exact same learning rate to every parameter**, which breaks down badly whenever different parameters have very different sensitivities — the real "ravine" experiment in that document showed a single learning rate forced into an impossible trade-off between zigzagging in a steep direction and crawling in a shallow one. RMSprop belongs to a family of optimizers that fixes this by giving **each parameter its own, individually-adjusted learning rate**. To understand RMSprop, it helps to first understand **AdaGrad** — the earlier optimizer it directly improves on.

---

## 2. AdaGrad: Giving Every Parameter Its Own Learning Rate

AdaGrad's idea: keep a running record of how large each parameter's gradients have been so far, and use that record to shrink the learning rate for parameters that have consistently had large gradients — while leaving parameters with small, infrequent gradients (common in sparse data) closer to their original learning rate.

```
G_t = G_{t-1} + g_t²                    (accumulate squared gradients)
w ← w − (η / √(G_t + ε))·g_t             (scale the learning rate down by that accumulator)
```

![AdaGrad vs. RMSprop formulas](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/adagrad_rmsprop_formulas.png)

Since `G_t` grows whenever a parameter has a large gradient, dividing by `√(G_t + ε)` automatically shrinks that parameter's effective step size — exactly the per-parameter adaptivity that plain gradient descent lacks.

---

## 3. AdaGrad's Fatal Flaw: The Vanishing Learning Rate

`G_t` is a **running sum** — and a sum of squared numbers can only ever grow (or stay flat), never shrink. Every single gradient observed over the entire training run, no matter how long ago, keeps contributing to `G_t` forever, and it never resets or decays. As training goes on and on, `G_t` keeps climbing, `√(G_t + ε)` keeps growing, and the effective learning rate `η / √(G_t + ε)` keeps shrinking — eventually becoming so small that updates are negligible, **even if the model hasn't actually reached a good solution yet**. This is exactly the flaw the video describes: AdaGrad can stop making meaningful progress well before convergence, purely because of how much history has piled up in its accumulator.

---

## 4. Real Experiment: Watching the Accumulator Grow

Training an actual small neural network on a real classification task for 3000 epochs, tracking the mean value of AdaGrad's accumulator alongside RMSprop's (introduced properly in the next section) for the same hidden layer:

![Real experiment: how the accumulator behaves over 3000 epochs](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/accumulator_growth_real.png)

*AdaGrad's accumulator climbs steadily for the entire 3000-epoch run, with no sign of leveling off — exactly the unbounded growth the formula predicts. RMSprop's accumulator, by contrast, stays close to zero throughout, since (as the next section explains) it forgets old gradients rather than permanently accumulating them.*

---

## 5. RMSprop's Fix: an EWMA Instead of a Sum

RMSprop makes exactly **one** change to AdaGrad: replace the running sum with an **exponentially weighted moving average** (see the [EWMA notes](ewma.md) for the full derivation of this formula):

```
G_t = β·G_{t-1} + (1−β)·g_t²             (EWMA of squared gradients, not a running sum)
w ← w − (η / √(G_t + ε))·g_t              (the update rule itself is UNCHANGED)
```

Because an EWMA continuously "forgets" old values at a rate controlled by `β` (typically 0.9), `G_t` no longer grows without bound — it settles into an equilibrium that reflects how large *recent* gradients have been, not the total accumulated history since the start of training. This one change is the entire difference between AdaGrad and RMSprop, and it directly targets the failure mode from Section 3.

---

## 6. Real Experiment: Does the Effective Learning Rate Actually Stay Healthy?

Tracking each optimizer's actual effective per-weight step size (`η / √(G_t + ε)`) over the same 3000-epoch run:

![Real experiment: does the learning rate actually vanish?](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/effective_lr_real.png)

*AdaGrad's effective learning rate falls monotonically for the entire run, from about 78 down to about 12 — over 6× smaller by the end, with no indication it will stop shrinking. RMSprop's effective learning rate, after an initial rise, stabilizes and fluctuates around a healthy range (roughly 60–70) for the rest of training — neither vanishing nor exploding, adapting to recent gradient sizes rather than the full accumulated history.*

---

## 7. Real Experiment: A Full Training Comparison

Putting it together: the same network, same data, trained with plain SGD, AdaGrad, and RMSprop:

![Real experiment: AdaGrad's fast start, then a stall](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/training_comparison_real.png)

*AdaGrad is genuinely the fastest optimizer early in training — its aggressive per-parameter scaling pays off immediately. But the zoomed inset reveals what happens next: AdaGrad's loss essentially freezes around 0.137 for the remaining ~2800 epochs, while RMSprop, starting from the same point, keeps steadily improving all the way down to 0.122. This is precisely the "stops converging before reaching the minimum" problem described in the video, measured directly rather than just asserted.*

---

## 8. Summary Table

![Plain GD vs. AdaGrad vs. RMSprop](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/comparison_table.png)

As the video notes, RMSprop has few major drawbacks and remains one of the most reliable and widely used optimizers, competitive even with more recent methods like Adam (which, as previewed in the [EWMA notes](ewma.md), combines RMSprop's approach with Momentum).

---

## 9. Practical Implementation in Keras

```python
from tensorflow import keras

model = keras.Sequential([...])

# AdaGrad -- rarely the best choice for long training runs, given Section 3-4/7
model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.01), loss="binary_crossentropy")

# RMSprop -- the standard fix
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9), loss="binary_crossentropy")
```

Keras's `rho` parameter is exactly `β` in the notation used throughout this document.

---

## 10. Code: AdaGrad and RMSprop From Scratch

```python
import numpy as np

def adagrad_update(w, grad, G, lr, eps=1e-8):
    """AdaGrad: G accumulates forever (a running sum)."""
    G += grad ** 2
    w -= (lr / np.sqrt(G + eps)) * grad
    return w, G

def rmsprop_update(w, grad, G, lr, beta=0.9, eps=1e-8):
    """RMSprop: G is an EWMA -- the one change from AdaGrad."""
    G = beta * G + (1 - beta) * grad ** 2
    w -= (lr / np.sqrt(G + eps)) * grad
    return w, G

# ---------------------------------------------------------
# A minimal illustration: 50 identical (large, persistent) gradients
# in a row -- shows AdaGrad's accumulator growing every step while
# RMSprop's settles into equilibrium
# ---------------------------------------------------------
grad = 2.0  # a constant, non-vanishing gradient (mimicking real mini-batch noise)
G_ada, G_rms = 0.0, 0.0
w_ada, w_rms = 0.0, 0.0

for step in range(50):
    w_ada, G_ada = adagrad_update(w_ada, grad, G_ada, lr=1.0)
    w_rms, G_rms = rmsprop_update(w_rms, grad, G_rms, lr=1.0)

print(f"After 50 steps of a constant gradient:")
print(f"  AdaGrad accumulator G = {G_ada:.2f}  (keeps growing every step)")
print(f"  RMSprop accumulator G = {G_rms:.2f}  (settled at grad^2 = {grad**2:.2f})")
```

Running this: AdaGrad's `G` reaches `50 × 2.0² = 200.0` after 50 steps of a constant gradient (and would reach `5000 × 4 = 20000` after 5000 steps — unbounded), while RMSprop's `G` converges to exactly `grad² = 4.0`, its stable equilibrium value, regardless of how many more steps are taken.

---

## 11. Key Takeaways

- AdaGrad gives every parameter its own learning rate by dividing by the square root of that parameter's **accumulated squared gradients**, `G_t = G_{t-1} + g_t²`.
- Because `G_t` is a running sum, it only ever grows, causing the effective learning rate to shrink toward zero over long training runs — a real experiment confirmed this directly, with the effective learning rate falling from ~78 to ~12 over 3000 epochs.
- This causes AdaGrad to genuinely **stop improving** well before convergence — a real training run showed AdaGrad's loss freezing at 0.137 for roughly 2800 epochs while another optimizer, given the exact same starting point, kept improving.
- **RMSprop's entire fix** is replacing that running sum with an **EWMA** (the same formula from the [EWMA notes](ewma.md)): `G_t = β·G_{t-1} + (1−β)·g_t²`. Because an EWMA forgets old values, `G_t` settles into a stable equilibrium instead of growing forever.
- A real experiment confirmed RMSprop's effective learning rate stabilizes in a healthy range rather than vanishing, and the resulting network kept improving to a meaningfully better loss (0.122 vs. AdaGrad's frozen 0.137) over the same training budget.
- RMSprop has few serious drawbacks and remains one of the most trusted and widely used optimizers in deep learning today.

---

## 12. Further Reading

- Duchi, J., Hazan, E., & Singer, Y. (2011). *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization* — the original AdaGrad paper.
- Hinton, G. (2012). *Neural Networks for Machine Learning*, Lecture 6e (Coursera) — RMSprop was introduced here rather than in a formal paper, and this lecture remains the primary original source.
- See also this repo's [`ewma.md`](ewma.md) (the exact formula RMSprop's accumulator is built from) and [`optimizers-part1.md`](optimizers-part1.md) (the ravine and static-learning-rate problems that motivate per-parameter adaptive methods in the first place).

---

*Diagrams in this document were generated programmatically — including real trained-network experiments measuring accumulator growth, effective learning rate, and a full training comparison between SGD, AdaGrad, and RMSprop — and are hosted in this repo's [`images/`](https://github.com/versatilerajan/deepcontent/tree/main/images) folder.*
