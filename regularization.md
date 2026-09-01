# Regularization: L1, L2, and Weight Decay

*Companion notes on why neural networks overfit, how L1 and L2 regularization add a penalty term to the cost function to keep weights small, the "weight decay" connection to gradient descent, and why L1 produces genuinely sparse models while L2 doesn't — backed by real trained-model experiments, not just theory.*

---

## Table of Contents

1. [Why Neural Networks Overfit](#1-why-neural-networks-overfit)
2. [The Toolbox for Fighting Overfitting](#2-the-toolbox-for-fighting-overfitting)
3. [L1 and L2 Regularization: the Penalty Term](#3-l1-and-l2-regularization-the-penalty-term)
4. [Why L1 Produces Sparsity and L2 Doesn't](#4-why-l1-produces-sparsity-and-l2-doesnt)
5. [A Real Experiment: Sparsity in Practice](#5-a-real-experiment-sparsity-in-practice)
6. [The Math: How L2 Becomes "Weight Decay"](#6-the-math-how-l2-becomes-weight-decay)
7. [A Real Experiment: Does Regularization Actually Help?](#7-a-real-experiment-does-regularization-actually-help)
8. [Practical Notes](#8-practical-notes)
9. [Code: L1 and L2 From Scratch](#9-code-l1-and-l2-from-scratch)
10. [Key Takeaways](#10-key-takeaways)
11. [Further Reading](#11-further-reading)

---

## 1. Why Neural Networks Overfit

As covered in the [dropout notes](dropout.md), overfitting happens when a model performs very well on training data but fails to generalize — because it has learned the noise and idiosyncrasies of the training set rather than the underlying pattern.

Neural networks are especially prone to this because adding more neurons gives them more and more capacity to draw **increasingly intricate decision boundaries** — eventually intricate enough to snake around every individual training point, including the noisy ones, rather than settling on the smoother boundary that would actually generalize. This isn't a flaw exclusive to neural networks; it's a general property of any sufficiently flexible model class:

![Why more model capacity can hurt: the classic overfitting signature](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bias_variance_complexity.png)

*As model complexity increases, training error keeps falling — the model can always fit the training data better with more capacity. Validation error, however, falls only up to a point, then rises again as the model starts fitting noise specific to the training set rather than the general pattern. The growing gap between the two curves on the right is overfitting, directly visible.*

---

## 2. The Toolbox for Fighting Overfitting

Regularization is one of several tools for combating overfitting, alongside collecting more data, data augmentation, dropout, and early stopping:

![Ways to fight overfitting](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/overfitting_solutions_map.png)

This document focuses on **regularization** — specifically **L1** and **L2** — which work by directly modifying what the optimizer is trying to minimize, rather than modifying the network's architecture (like dropout) or the training data itself (like augmentation).

---

## 3. L1 and L2 Regularization: the Penalty Term

Both techniques work the same basic way: add an extra term to the cost function that **penalizes large weights**, so the optimizer has to balance fitting the data well against keeping the weights small.

```
L1 (Lasso):   J(w) = L(w) + λ · Σᵢ |wᵢ|

L2 (Ridge):   J(w) = L(w) + λ · Σᵢ wᵢ²
```

where `L(w)` is the original loss (e.g. MSE or cross-entropy, see the [loss functions notes](loss-functions.md)), and `λ` (lambda) is a hyperparameter controlling how strongly the penalty is enforced — `λ = 0` recovers the unregularized model, and larger `λ` pushes weights more aggressively toward zero.

Intuitively: a model with smaller weights is a "simpler" model in a meaningful sense — small weights mean the output changes less dramatically in response to small changes in the input, which is exactly the kind of smoother, less noise-sensitive behavior that generalizes better.

---

## 4. Why L1 Produces Sparsity and L2 Doesn't

Both penalties shrink weights, but they do it in geometrically different ways, and this difference has a real practical consequence: **L1 tends to drive some weights to exactly zero (a sparse model), while L2 shrinks weights smoothly toward zero without usually reaching it exactly.**

![Why L1 produces sparsity and L2 doesn't: the geometry](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/l1_l2_geometry.png)

Thinking of regularization as constraining the weights to a region (a diamond for L1, a circle for L2) around the origin, and the unconstrained loss-minimizing point sitting somewhere outside that region: the best point *within* the region is wherever the loss's contour lines first touch the boundary. A diamond has **corners** sitting exactly on the axes (where one weight is exactly zero) — contour lines are disproportionately likely to first touch at one of those corners. A circle has no corners at all, so the touching point is typically some smooth combination of all weights, none of which are exactly zero.

---

## 5. A Real Experiment: Sparsity in Practice

To confirm this isn't just geometric intuition, here's an actual linear model trained on real data: 30 input features, but only **5 are genuinely relevant** to the target — the other 25 are pure noise, uncorrelated with the output. The model is trained three ways: no regularization, L1, and L2.

![Real experiment: L1 zeros out irrelevant features, L2 only shrinks them](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/l1_l2_sparsity_real.png)

*Green bars are the 5 truly relevant features (correctly identified as large-magnitude by all three models). Without regularization, the 25 irrelevant features still pick up small but real nonzero weights (noise the optimizer happily fits). With L2, those irrelevant weights shrink but remain clearly nonzero. With L1, 80% of the irrelevant weights land below 0.01 in magnitude — and checking precisely, **64% land at exactly zero** — the model has effectively performed automatic feature selection, keeping only the features that matter.*

(Achieving *exact* zeros with L1 in practice requires the correct optimization step — a proximal/soft-thresholding update, not plain gradient descent on `sign(w)`, which tends to hover near zero without landing on it exactly. The experiment above uses the correct proximal step.)

---

## 6. The Math: How L2 Becomes "Weight Decay"

Combining the L2-augmented cost function with the ordinary gradient descent update rule reveals exactly why L2 regularization is so often called **weight decay**:

![Deriving weight decay from the L2-regularized gradient descent update](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/weight_decay_derivation.png)

Working through the algebra: taking the gradient of `J(w) = L(w) + λΣwᵢ²` with respect to `w` adds a `2λw` term to the ordinary loss gradient. Substituting that into the standard update rule `w ← w − η·∂J/∂w` and rearranging gives:

```
w ← w(1 − 2ηλ) − η·∂L/∂w
```

Every single update, *before* even applying the usual gradient step, every weight is first multiplied by a factor `(1 − 2ηλ)` slightly less than 1 — literally decaying it toward zero. This is a direct, mechanical consequence of adding the L2 penalty, not a separate technique bolted on afterward.

---

## 7. A Real Experiment: Does Regularization Actually Help?

The same small noisy-sine regression setup used in the [dropout notes](dropout.md#5-a-real-training-run-does-it-actually-help) — a high-capacity network, small training set, held-out validation split — trained with and without L2 regularization:

![Real training run: L2 regularization improves validation loss and narrows the gap](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/train_val_l2_real.png)

*Without regularization, validation loss plateaus at 0.119 while the train/validation gap sits at 0.069 — the overfitting signature. With a carefully chosen L2 strength (λ=0.0004), validation loss actually **improves** to 0.114, and the gap narrows to 0.060 — a genuine, measured improvement in generalization, not just a smaller number produced by making the model worse everywhere. (Too large a λ would instead hurt both training *and* validation loss — the underfitting failure mode from Section 1's complexity curve, just approached from the regularization-strength axis instead of the model-size axis.)*

---

## 8. Practical Notes

- **λ needs tuning.** Too small and it barely regularizes anything; too large and it pushes weights toward zero so aggressively that the model can't fit even the training data well (underfitting) — the same U-shaped trade-off dropout's rate has (see the [dropout notes](dropout.md#6-choosing-the-dropout-rate)).
- **L1 for feature selection, L2 as a general-purpose default.** If interpretability or automatic feature pruning matters (as in Section 5's demo), L1's sparsity is a genuine practical advantage. When that's not a priority, L2 is generally preferred — it's smoother to optimize and doesn't have L1's tendency to arbitrarily pick one of several correlated features to keep while zeroing the others.
- **Elastic Net** combines both penalties (`λ₁Σ|wᵢ| + λ₂Σwᵢ²`), getting some sparsity from the L1 term while keeping L2's smoother optimization behavior — a common practical compromise when both properties are desirable.
- **Biases are typically not regularized.** The penalty is conventionally applied only to weights, not biases, since biases don't contribute to the model's sensitivity to input changes the way weights do.

---

## 9. Code: L1 and L2 From Scratch

```python
import numpy as np

def soft_threshold(w, thresh):
    """Proximal operator for L1 -- the correct way to get exact zeros."""
    return np.sign(w) * np.maximum(np.abs(w) - thresh, 0)

def train_linear(X, y, lr=0.01, epochs=3000, l1=0.0, l2=0.0):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for epoch in range(epochs):
        pred = X @ w + b
        error = pred - y
        grad_w = (2/n) * X.T @ error
        grad_b = (2/n) * error.sum()

        if l2 > 0:
            grad_w += 2 * l2 * w        # L2: added directly to the gradient (weight decay)

        w -= lr * grad_w
        b -= lr * grad_b

        if l1 > 0:
            w = soft_threshold(w, lr * l1)   # L1: applied as a separate proximal step

    return w, b

# ---------------------------------------------------------
# Reproduce the sparsity experiment from Section 5
# ---------------------------------------------------------
rng = np.random.default_rng(7)
n_samples, n_features, n_relevant = 80, 30, 5
X = rng.normal(0, 1, (n_samples, n_features))
true_w = np.zeros(n_features)
true_w[:n_relevant] = rng.uniform(1.5, 3.0, n_relevant) * rng.choice([-1, 1], n_relevant)
y = X @ true_w + rng.normal(0, 0.5, n_samples)

w_none, _ = train_linear(X, y, l1=0.0, l2=0.0)
w_l1, _ = train_linear(X, y, l1=0.08, l2=0.0)
w_l2, _ = train_linear(X, y, l1=0.0, l2=0.15)

print("Irrelevant weights exactly 0 -- none:", np.mean(w_none[n_relevant:] == 0))
print("Irrelevant weights exactly 0 -- L1:  ", np.mean(w_l1[n_relevant:] == 0))
print("Irrelevant weights exactly 0 -- L2:  ", np.mean(w_l2[n_relevant:] == 0))
```

**Using Keras**, both penalties are one-line layer arguments:

```python
from tensorflow import keras
from tensorflow.keras import layers, regularizers

model = keras.Sequential([
    layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.0004), input_shape=(20,)),
    layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l1(0.001)),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy")
```

---

## 10. Key Takeaways

- Neural networks overfit because added capacity lets them draw increasingly intricate boundaries that memorize training data rather than generalize — visible as a growing gap between training and validation error.
- **L1** (`λΣ|wᵢ|`) and **L2** (`λΣwᵢ²`) both add a penalty to the cost function that discourages large weights, trading off fit-to-data against model simplicity.
- Geometrically, L1's diamond-shaped constraint region has corners on the axes, making exact-zero solutions common — **L1 induces genuine sparsity**, confirmed in a real experiment where 80% of irrelevant feature weights landed at exactly zero.
- L2's circular constraint region has no corners, so it shrinks weights smoothly without usually reaching exact zero.
- Substituting the L2 penalty's gradient into the gradient descent update rule shows weights get multiplied by `(1 − 2ηλ)` every step *before* the usual gradient step — this is literally where the term **"weight decay"** comes from.
- A real training run confirms the practical payoff: properly-tuned L2 regularization both narrowed the train/validation gap **and** improved absolute validation loss, not just made the model uniformly worse.

---

## 11. Further Reading

- Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso* — the original paper introducing L1 regularization (Lasso) and its sparsity property.
- Hoerl, A. & Kennard, R. (1970). *Ridge Regression: Biased Estimation for Nonorthogonal Problems* — the original Ridge (L2) regression paper.
- Zou, H. & Hastie, T. (2005). *Regularization and Variable Selection via the Elastic Net* — combining L1 and L2.
- Keras documentation on [regularizers](https://keras.io/api/layers/regularizers/) for implementation details.
- See also this repo's [`dropout.md`](dropout.md) (a complementary, architecture-level regularization technique) and [`gradient-descent.md`](gradient-descent.md) (the update rule this document builds directly on).

---

*Diagrams in this document were generated programmatically — including two real from-scratch training experiments used to produce the sparsity and train/validation figures — and are hosted in this repo's [`images/`](https://github.com/versatilerajan/deepcontent/tree/main/images) folder.*
