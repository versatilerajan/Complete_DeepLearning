# Loss Functions in Deep Learning: Formulas, Intuition, and Examples

*Companion notes on why loss functions exist, the distinction between loss and cost, and a tour of the loss functions used across regression and classification tasks — MSE, MAE, Huber, Binary Cross-Entropy, Categorical Cross-Entropy, and Sparse Categorical Cross-Entropy.*

---

## Table of Contents

1. [What Is a Loss Function?](#1-what-is-a-loss-function)
2. [Loss vs. Cost](#2-loss-vs-cost)
3. [Mean Squared Error (MSE)](#3-mean-squared-error-mse)
4. [Mean Absolute Error (MAE)](#4-mean-absolute-error-mae)
5. [Worked Example: MSE vs. MAE on an Outlier](#5-worked-example-mse-vs-mae-on-an-outlier)
6. [Huber Loss](#6-huber-loss)
7. [Binary Cross-Entropy](#7-binary-cross-entropy)
8. [Categorical Cross-Entropy](#8-categorical-cross-entropy)
9. [Sparse Categorical Cross-Entropy](#9-sparse-categorical-cross-entropy)
10. [Which Loss for Which Task?](#10-which-loss-for-which-task)
11. [Code: Computing Every Loss From Scratch](#11-code-computing-every-loss-from-scratch)
12. [Key Takeaways](#12-key-takeaways)
13. [Further Reading](#13-further-reading)

---

## 1. What Is a Loss Function?

A **loss function** is a mathematical way to measure how wrong a single prediction is — the gap between what the model predicted (`ŷ`) and what actually happened (`y`). It converts "how good is this prediction" into a single number that gradient descent can then try to minimize by adjusting the model's weights and biases.

```
L = L(y, ŷ)
```

The specific formula for `L` depends entirely on the task: regression problems (predicting a continuous number) and classification problems (predicting a category) need fundamentally different notions of "distance" between a prediction and the truth, which is why so many different loss functions exist.

---

## 2. Loss vs. Cost

These two terms are often used loosely, but there's a precise, useful distinction:

- **Loss** — the error for a **single training example**.
- **Cost** — the **average** loss over the **entire training set** (or a mini-batch).

```
Loss (per example):     Lₙ = L(yₙ, ŷₙ)

Cost (whole dataset):   J(w, b) = (1/m) · Σₙ Lₙ
```

where `m` is the number of training examples. Gradient descent minimizes the **cost** `J`, since that's what tells you how well the model is doing overall — but it's computed by summing/averaging the **loss** at every individual data point.

![Loss (per example) vs. cost (averaged over the dataset)](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/loss_vs_cost.png)

*Five individual predictions each get their own loss value `Lₙ` (here, squared error). Averaging all five gives the cost `J` for the whole dataset — the single number training actually tries to minimize.*

---

## 3. Mean Squared Error (MSE)

The default loss for **regression** tasks — predicting a continuous number.

```
Loss (per example):  L = (y − ŷ)²

Cost (MSE):          J = (1/m) · Σₙ (yₙ − ŷₙ)²
```

- Easy to interpret and differentiate everywhere (smooth, no sharp corners).
- Because the error is **squared**, large errors are penalized disproportionately more than small ones — a prediction that's off by 10 contributes 100× more loss than one off by 1. This makes MSE **sensitive to outliers**: a single badly-predicted point can dominate the entire cost function.

---

## 4. Mean Absolute Error (MAE)

Also for **regression**, but penalizes errors linearly instead of quadratically.

```
Loss (per example):  L = |y − ŷ|

Cost (MAE):          J = (1/m) · Σₙ |yₙ − ŷₙ|
```

- Much more **robust to outliers** — a huge error contributes proportionally to the cost rather than exploding quadratically.
- Downside: `|y − ŷ|` has a sharp corner (is **not differentiable**) exactly at `y = ŷ`, which can make optimization slightly less well-behaved right around zero error compared to MSE's perfectly smooth bowl shape.

![MSE vs MAE loss curves](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/mse_mae_curves.png)

*MSE (red) curves upward quadratically — a residual of 4 contributes a loss of 16. MAE (blue) grows in a straight line — the same residual of 4 contributes a loss of only 4. This single difference in shape is the entire story behind MSE's outlier-sensitivity vs. MAE's robustness.*

---

## 5. Worked Example: MSE vs. MAE on an Outlier

Consider 5 predictions, where the 5th happens to be a bad outlier prediction:

| Point | Actual y | Predicted ŷ | Error e = y−ŷ | e² | \|e\| |
|---|---|---|---|---|---|
| 1 | 3 | 2.5 | 0.5 | 0.25 | 0.5 |
| 2 | 5 | 5.2 | −0.2 | 0.04 | 0.2 |
| 3 | 2 | 2.1 | −0.1 | 0.01 | 0.1 |
| 4 | 8 | 7.8 | 0.2 | 0.04 | 0.2 |
| 5 | 50 | 10 | 40 | 1600 | 40 |

```
MSE = (0.25 + 0.04 + 0.01 + 0.04 + 1600) / 5   = 320.07
MAE = (0.5  + 0.2  + 0.1  + 0.2  + 40)   / 5   = 8.20
```

![One outlier dominates MSE far more than MAE](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/outlier_sensitivity.png)

*Points 1–4 are all predicted almost perfectly, yet MSE = 320.07 — a number that makes the model look far worse than it actually is on 4 out of 5 points, entirely because of point 5's squared error of 1600. MAE = 8.20 tells a much more representative story of typical performance.*

---

## 6. Huber Loss

A hybrid designed to get the best of both: smooth and well-behaved near zero (like MSE), but robust to outliers far from zero (like MAE). It introduces one hyperparameter, `δ` (delta), that sets where the switch happens:

```
         { 0.5·(y − ŷ)²                if |y − ŷ| ≤ δ
L_δ  =   {
         { δ·(|y − ŷ| − 0.5·δ)         if |y − ŷ| > δ
```

![Huber loss combines MSE near zero and MAE farther out](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/huber_loss.png)

*Inside the `±δ` band, Huber loss is the smooth MSE-style curve. Outside that band, it switches to the MAE-style straight line — so large outliers only ever contribute linearly, not quadratically, while small errors near zero still get the smooth, well-conditioned gradient that MSE provides.*

**Applying Huber (δ=1) to the same outlier example from Section 5:** point 5's contribution becomes `1×(40 − 0.5) = 39.5` instead of MSE's `1600` — dramatically reduced, while points 1–4 (all with small errors) are barely affected. The resulting average Huber cost is `7.93`, much closer to MAE's robust `8.20` than to MSE's outlier-dominated `320.07`.

---

## 7. Binary Cross-Entropy

The standard loss for **binary classification** (e.g., spam/not-spam, yes/no). Requires the output layer to use a **sigmoid** activation, so `ŷ` is a probability between 0 and 1.

```
L = −[ y·log(ŷ) + (1 − y)·log(1 − ŷ) ]
```

Only one of the two terms is ever "active" for a given example, since `y` is either 0 or 1:

- If `y = 1`: loss simplifies to `−log(ŷ)` — penalizes the model for assigning *low* probability to the true class.
- If `y = 0`: loss simplifies to `−log(1 − ŷ)` — penalizes the model for assigning *high* probability when it shouldn't have.

**Worked example:** true label `y = 1`.
- Confident and correct: `ŷ = 0.9` → `L = −log(0.9) ≈ 0.105` (small loss)
- Confident and wrong: `ŷ = 0.1` → `L = −log(0.1) ≈ 2.303` (large loss)

![Binary cross-entropy loss curve](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bce_curve.png)

*As a prediction gets closer to being confidently correct, loss shrinks toward 0. As it gets confidently wrong, loss blows up toward infinity — this steep penalty for confident mistakes is what makes cross-entropy such an effective training signal for classifiers.*

---

## 8. Categorical Cross-Entropy

The generalization of binary cross-entropy to **multi-class classification** (more than 2 classes). Requires:

- The true label `y` to be **one-hot encoded** — a vector with a `1` in the correct class's position and `0` elsewhere.
- The output layer to use a **softmax** activation, so the model's output `ŷ` is a full probability distribution over all classes (all entries positive, summing to 1).

```
L = − Σ_c  y_c · log(ŷ_c)
```

summed over every class `c`. Because `y_c` is `0` for every class except the true one, **every term except the true class's vanishes** — the formula collapses to just `−log(ŷ_true class)`.

![Categorical cross-entropy with one-hot labels and softmax output](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/multiclass_crossentropy.png)

*True label "dog" is one-hot encoded as `[0, 1, 0]`. The softmax output is `[0.2, 0.7, 0.1]`. Only the "dog" entry (0.7) ever enters the loss: `L = −log(0.7) ≈ 0.357`.*

---

## 9. Sparse Categorical Cross-Entropy

Mathematically **identical** to categorical cross-entropy — same formula, same result — but with a more efficient way of specifying the true label: as a single **integer class index** (e.g., `1` for "dog") instead of a full one-hot vector (`[0, 1, 0]`).

This matters purely for **efficiency**: with, say, 10,000 classes, storing a one-hot vector per example wastes memory on 9,999 zeros. Sparse categorical cross-entropy skips building that vector entirely and just looks up `ŷ` at the given integer index directly:

```
L = −log(ŷ_[true class index])
```

Using the same example as Section 8 (true class index `1`, softmax output `[0.2, 0.7, 0.1]`): `L = −log(0.7) ≈ 0.357` — exactly the same number, just without ever materializing `[0, 1, 0]`.

---

## 10. Which Loss for Which Task?

| Loss | Task | Output activation | Notes |
|---|---|---|---|
| MSE | Regression | Linear (none) | Sensitive to outliers |
| MAE | Regression | Linear (none) | Robust to outliers, non-differentiable at 0 |
| Huber | Regression | Linear (none) | Best of both, needs tuning δ |
| Binary Cross-Entropy | Binary classification | Sigmoid | Penalizes confident wrong answers heavily |
| Categorical Cross-Entropy | Multi-class classification | Softmax | Needs one-hot labels |
| Sparse Categorical Cross-Entropy | Multi-class classification | Softmax | Needs integer labels; more memory-efficient |

---

## 11. Code: Computing Every Loss From Scratch

```python
import numpy as np

# ---------------------------------------------------------
# Regression losses
# ---------------------------------------------------------
def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))

def huber(y, y_hat, delta=1.0):
    e = y - y_hat
    is_small = np.abs(e) <= delta
    squared_term = 0.5 * e ** 2
    linear_term = delta * (np.abs(e) - 0.5 * delta)
    return np.mean(np.where(is_small, squared_term, linear_term))

# ---------------------------------------------------------
# Classification losses
# ---------------------------------------------------------
def binary_cross_entropy(y, y_hat, eps=1e-12):
    y_hat = np.clip(y_hat, eps, 1 - eps)  # avoid log(0)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def categorical_cross_entropy(y_onehot, y_hat, eps=1e-12):
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(np.sum(y_onehot * np.log(y_hat), axis=-1))

def sparse_categorical_cross_entropy(y_idx, y_hat, eps=1e-12):
    y_hat = np.clip(y_hat, eps, 1 - eps)
    true_class_probs = y_hat[np.arange(len(y_idx)), y_idx]
    return -np.mean(np.log(true_class_probs))

# ---------------------------------------------------------
# Reproduce the worked examples from this document
# ---------------------------------------------------------
y = np.array([3, 5, 2, 8, 50])
y_hat = np.array([2.5, 5.2, 2.1, 7.8, 10])
print("MSE:  ", mse(y, y_hat))          # 320.068
print("MAE:  ", mae(y, y_hat))          # 8.2
print("Huber:", huber(y, y_hat, 1.0))   # 7.934

y_bin = np.array([1])
print("BCE (confident+correct):", binary_cross_entropy(y_bin, np.array([0.9])))  # 0.105
print("BCE (confident+wrong):  ", binary_cross_entropy(y_bin, np.array([0.1])))  # 2.303

y_onehot = np.array([[0, 1, 0]])
y_softmax = np.array([[0.2, 0.7, 0.1]])
print("Categorical CE:", categorical_cross_entropy(y_onehot, y_softmax))         # 0.357

y_idx = np.array([1])  # same "dog" example, integer-encoded
print("Sparse Categorical CE:", sparse_categorical_cross_entropy(y_idx, y_softmax))  # 0.357
```

**Using Keras/TensorFlow** (production-grade versions of the same formulas):

```python
import tensorflow as tf

model.compile(optimizer="adam", loss="mse")                              # regression
model.compile(optimizer="adam", loss="mae")                              # regression, outlier-robust
model.compile(optimizer="adam", loss=tf.keras.losses.Huber(delta=1.0))   # regression, hybrid
model.compile(optimizer="adam", loss="binary_crossentropy")              # binary classification
model.compile(optimizer="adam", loss="categorical_crossentropy")         # multi-class, one-hot labels
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")  # multi-class, integer labels
```

---

## 12. Key Takeaways

- A **loss** measures the error of one prediction; a **cost** is the average loss over the whole training set — gradient descent minimizes the cost.
- **MSE** is smooth and easy to optimize but heavily punishes (and is distorted by) outliers, since errors are squared.
- **MAE** is robust to outliers since errors grow linearly, but has a non-differentiable kink at zero error.
- **Huber loss** blends both: quadratic (MSE-like) near zero, linear (MAE-like) beyond a threshold `δ` — a practical default when data may contain both normal points and outliers.
- **Binary cross-entropy** (with sigmoid output) and **categorical cross-entropy** (with softmax output + one-hot labels) are the standard losses for binary and multi-class classification, both sharply penalizing confident wrong predictions.
- **Sparse categorical cross-entropy** is mathematically identical to categorical cross-entropy — it just accepts integer labels instead of one-hot vectors, saving substantial memory when there are many classes.

---

## 13. Further Reading

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning* (Chapter 5–6 cover loss functions and maximum likelihood estimation as their theoretical foundation).
- Huber, P. J. (1964). *Robust Estimation of a Location Parameter* — the original paper introducing what became Huber loss.
- Keras documentation on [losses](https://keras.io/api/losses/) for exact implementation details and numerical-stability tricks used in practice.
- See also this repo's [`perceptron.md`](perceptron.md) (introduces the perceptron loss, closely related to hinge loss) and [`mlp.md`](mlp.md) (where these losses plug into backpropagation).

---

*Diagrams in this document were generated programmatically to illustrate the concepts discussed above, and are hosted in this repo's [`images/`](https://github.com/versatilerajan/deepcontent/tree/main/images) folder.*
