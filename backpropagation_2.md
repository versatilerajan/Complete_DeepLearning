# Backpropagation, Part 2: Implementing Regression and Classification From Scratch

*Companion notes on the practical, code-first half of backpropagation — building a regression network and a classification network from scratch with NumPy, deriving the (surprisingly clean) classification gradients by hand, and comparing against the equivalent Keras implementation.*

---

## Table of Contents

1. [Recap: Where Part 1 Left Off](#1-recap-where-part-1-left-off)
2. [Regression: Training the Network Over Many Epochs](#2-regression-training-the-network-over-many-epochs)
3. [Classification: Adding Sigmoid and Binary Cross-Entropy](#3-classification-adding-sigmoid-and-binary-cross-entropy)
4. [Deriving the Classification Gradients](#4-deriving-the-classification-gradients)
5. [A Beautiful Simplification](#5-a-beautiful-simplification)
6. [A Practical Trap: Vanishing Gradients From Unscaled Inputs](#6-a-practical-trap-vanishing-gradients-from-unscaled-inputs)
7. [Training the Classification Network](#7-training-the-classification-network)
8. [Code: Both Networks, From Scratch](#8-code-both-networks-from-scratch)
9. [Comparing to Keras](#9-comparing-to-keras)
10. [Key Takeaways](#10-key-takeaways)
11. [What's Next](#11-whats-next)

---

## 1. Recap: Where Part 1 Left Off

[Part 1](backprop-part1.md) hand-derived the chain rule for a small **linear** network (2 inputs → 2 hidden neurons → 1 output, no activation functions) predicting a student's Placement Score from CGPA and Skills Score, and manually walked through a single epoch of updates.

Part 2 turns that into actual running code, in two stages:

1. **Regression** — the same linear network from Part 1, but now trained for many epochs to convergence.
2. **Classification** — the same architecture, but with a **sigmoid** activation added at every layer and **Binary Cross-Entropy** as the loss, predicting whether a student gets **placed** (0/1) instead of a continuous score.

---

## 2. Regression: Training the Network Over Many Epochs

Using the exact network and dataset from Part 1 (CGPA, Skills Score → Placement Score; weights initialized to 1, biases to 0; learning rate 0.001), running the same forward → loss → backward → update loop for **300 epochs** instead of just one:

![Regression network loss over 300 epochs](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/regression_training_curve.png)

*Almost the entire loss reduction happens in the very first couple of epochs (784 → under 1), exactly as Part 1's hand computation showed — the rest of training is a slow, steady refinement. After 300 epochs the loss settles around 0.28, with final predictions of 4.71, 5.15, and 5.60 against actual values of 4, 5, and 6.*

This is the same mechanics as Part 1, just automated into a loop and run to convergence rather than stopped after one pass.

---

## 3. Classification: Adding Sigmoid and Binary Cross-Entropy

For **classification** — predicting whether a student gets **placed** (`y ∈ {0, 1}`) — two things change:

- Every neuron (hidden **and** output) now applies a **sigmoid** activation, so its output is squashed into `(0, 1)`.
- The loss switches from MSE to **Binary Cross-Entropy**, matching what a probability output actually calls for (see the [loss functions notes](loss-functions.md) for the full derivation of why).

![Classification network: sigmoid at hidden and output layers](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/classification_network.png)

```
O1k = σ(w1k¹·x_i1 + w2k¹·x_i2 + b1k)     for k = 1, 2   (hidden neurons)
ŷ   = σ(w11²·O11 + w21²·O12 + b21)                       (output)

L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]
```

---

## 4. Deriving the Classification Gradients

The chain rule structure is identical to Part 1 — every gradient is still a product of local derivatives along the path from that parameter to the loss — but now there's an **extra factor at every layer**: the derivative of the sigmoid itself, `σ'(z) = σ(z)·(1−σ(z))`.

![Sigmoid activation and its derivative](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/sigmoid_and_derivative.png)

**Output layer** (`z21` is the pre-activation input to the output neuron, so `ŷ = σ(z21)`):

```
∂L/∂z21 = ∂L/∂ŷ · ∂ŷ/∂z21 = ŷ − y      (derived fully in Section 5 below)

∂L/∂w11² = ∂L/∂z21 · O11
∂L/∂w21² = ∂L/∂z21 · O12
∂L/∂b21  = ∂L/∂z21
```

**Hidden layer** — same structure as Part 1, but each hidden neuron's own sigmoid derivative now enters the chain:

```
∂L/∂w11¹ = ∂L/∂z21 · w11² · O11·(1−O11) · x_i1
∂L/∂w21¹ = ∂L/∂z21 · w11² · O11·(1−O11) · x_i2
∂L/∂b11  = ∂L/∂z21 · w11² · O11·(1−O11)

∂L/∂w12¹ = ∂L/∂z21 · w21² · O12·(1−O12) · x_i1
∂L/∂w22¹ = ∂L/∂z21 · w21² · O12·(1−O12) · x_i2
∂L/∂b12  = ∂L/∂z21 · w21² · O12·(1−O12)
```

Compare this to Part 1's purely linear gradients — the structure (weight × input, reusing the upstream gradient) is unchanged; the only addition is the `O1k·(1−O1k)` sigmoid-derivative factor at each hidden neuron.

---

## 5. A Beautiful Simplification

The `∂L/∂z21 = ŷ − y` result used above looks suspiciously simple for something built from a logarithm and a sigmoid — and that's not an accident.

![BCE + sigmoid simplification](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bce_sigmoid_simplification.png)

Working it out: `∂L/∂ŷ = −y/ŷ + (1−y)/(1−ŷ)`, and `∂ŷ/∂z21 = ŷ(1−ŷ)`. Multiplying these together:

```
∂L/∂z21 = [−y/ŷ + (1−y)/(1−ŷ)] · ŷ(1−ŷ)
        = −y(1−ŷ) + (1−y)ŷ
        = −y + yŷ + ŷ − yŷ
        = ŷ − y
```

Every messy log and sigmoid term cancels, leaving exactly `ŷ − y` — the **same simple error term** as MSE with a linear output (Part 1's `∂L/∂ŷ` was `−2(y−ŷ)`, which is just this same error up to a constant factor). This is precisely *why* sigmoid + Binary Cross-Entropy is the standard pairing for binary classification: the combination was chosen specifically because it produces this clean gradient, rather than needing sigmoid's derivative and BCE's derivative to be carried around separately through every subsequent computation.

---

## 6. A Practical Trap: Vanishing Gradients From Unscaled Inputs

Naively reusing Part 1's initialization (`weights=1`, `biases=0`) directly on raw features (CGPA=8, Skills=8) causes a real problem once sigmoid is involved. The pre-activation value reaching the first hidden neuron is `1×8 + 1×8 + 0 = 16` — and `σ(16) ≈ 0.9999999`, deep in the flat, saturated tail of the sigmoid curve.

![Vanishing gradient from unnormalized inputs](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/vanishing_gradient.png)

*With raw inputs, the hidden layer's gradient (`≈1.07×10⁻⁷`) is **six orders of magnitude smaller** than the output layer's gradient (`≈0.119`) — for practical purposes, the hidden layer stops learning almost entirely. This is the vanishing gradient problem in miniature: it isn't just a deep-network phenomenon, it can show up in a single hidden layer the moment inputs are large enough to saturate the activation.*

The fix used for the training run in the next section is the standard one: **scale the inputs** (here, dividing CGPA and Skills Score by 10) so pre-activation values land in the sensitive middle region of the sigmoid where gradients are actually informative.

---

## 7. Training the Classification Network

With inputs scaled (CGPA/10, Skills/10) and a slightly larger learning rate (0.1, since scaled-down inputs produce smaller gradients), training for 2000 epochs on 3 students (2 labeled "placed", 1 labeled "not placed"):

![Classification network training](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/classification_training_curve.png)

*Loss barely moves for the first few hundred epochs (the network is still escaping the initial saturation from Section 6), then drops steadily as the parameters move into a better-conditioned region. By epoch 2000, the network confidently predicts 0.998 and 0.904 for the two "placed" students, and 0.106 for the "not placed" student — all correctly separated by the 0.5 decision threshold.*

---

## 8. Code: Both Networks, From Scratch

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ---------------------------------------------------------
# REGRESSION (linear activations, MSE loss) — same as Part 1,
# just run for many epochs instead of one.
# ---------------------------------------------------------
def train_regression(X, y, epochs=300, lr=0.001):
    w1, b1 = np.ones((2, 2)), np.zeros(2)
    w2, b2 = np.ones((2, 1)), np.zeros(1)
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for x, yt in zip(X, y):
            O1 = x @ w1 + b1
            yhat = (O1 @ w2 + b2)[0]
            error = yt - yhat
            total_loss += error ** 2

            dL_dyhat = -2 * error
            dL_dw2 = dL_dyhat * O1.reshape(-1, 1)
            dL_db2 = dL_dyhat
            dL_dw1 = np.outer(x, dL_dyhat * w2.flatten())   # uses OLD w2 -- compute before updating it!
            dL_db1 = dL_dyhat * w2.flatten()                 # uses OLD w2

            w2 -= lr * dL_dw2
            b2 -= lr * dL_db2
            w1 -= lr * dL_dw1
            b1 -= lr * dL_db1
        losses.append(total_loss / len(X))
    return w1, b1, w2, b2, losses

# ---------------------------------------------------------
# CLASSIFICATION (sigmoid activations, Binary Cross-Entropy)
# ---------------------------------------------------------
def train_classification(X, y, epochs=2000, lr=0.1, eps=1e-12):
    w1, b1 = np.ones((2, 2)), np.zeros(2)
    w2, b2 = np.ones((2, 1)), np.zeros(1)
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for x, yt in zip(X, y):
            # forward pass
            z1 = x @ w1 + b1
            O1 = sigmoid(z1)
            z2 = (O1 @ w2 + b2)[0]
            yhat = sigmoid(z2)
            yhat_c = np.clip(yhat, eps, 1 - eps)
            total_loss += -(yt*np.log(yhat_c) + (1-yt)*np.log(1-yhat_c))

            # backward pass -- note the clean (yhat - y) simplification
            delta2 = yhat - yt
            dL_dw2 = delta2 * O1.reshape(-1, 1)
            dL_db2 = delta2
            delta1 = delta2 * w2.flatten() * O1 * (1 - O1)   # uses OLD w2 -- compute before updating it!
            dL_dw1 = np.outer(x, delta1)
            dL_db1 = delta1

            w2 -= lr * dL_dw2
            b2 -= lr * dL_db2
            w1 -= lr * dL_dw1
            b1 -= lr * dL_db1
        losses.append(total_loss / len(X))
    return w1, b1, w2, b2, losses

# ---------------------------------------------------------
# Reproduce this document's results
# ---------------------------------------------------------
X_reg = np.array([[8,8],[7,9],[6,10]], dtype=float)
y_reg = np.array([4,5,6], dtype=float)
*_, reg_losses = train_regression(X_reg, y_reg)
print("Regression final loss:", reg_losses[-1])          # ~0.28

X_clf = np.array([[8,8],[7,9],[6,10]], dtype=float) / 10.0   # scaled!
y_clf = np.array([1,1,0], dtype=float)
w1, b1, w2, b2, clf_losses = train_classification(X_clf, y_clf)
print("Classification final loss:", clf_losses[-1])

for x, yt in zip(X_clf, y_clf):
    O1 = sigmoid(x @ w1 + b1)
    yhat = sigmoid((O1 @ w2 + b2)[0])
    print(f"actual={yt}, predicted P(placed)={yhat:.4f}")
```

---

## 9. Comparing to Keras

The instructor's point in building this by hand is to demystify exactly what a framework like Keras is doing under the hood. Both networks above translate directly:

```python
from tensorflow import keras
from tensorflow.keras import layers

# Regression
reg_model = keras.Sequential([
    layers.Dense(2, activation="linear", input_shape=(2,)),
    layers.Dense(1, activation="linear"),
])
reg_model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001), loss="mse")
reg_model.fit(X_reg, y_reg, epochs=300, verbose=0)

# Classification
clf_model = keras.Sequential([
    layers.Dense(2, activation="sigmoid", input_shape=(2,)),
    layers.Dense(1, activation="sigmoid"),
])
clf_model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1), loss="binary_crossentropy")
clf_model.fit(X_clf, y_clf, epochs=2000, verbose=0)
```

Keras runs the exact same forward pass → loss → backward pass (via automatic differentiation rather than hand-derived formulas) → gradient descent update loop — it just handles the chain rule bookkeeping automatically, at arbitrary scale, across arbitrarily deep and wide networks. Having derived it by hand is what makes that automation legible rather than magical.

---

## 10. Key Takeaways

- The regression network from Part 1 needs no new theory to reach convergence — just running the same loop for more epochs.
- Classification requires two changes: sigmoid activations (to output values in `(0,1)`) and Binary Cross-Entropy loss (matched to those probability outputs).
- Every hidden-layer classification gradient picks up one extra factor compared to Part 1's linear case: the sigmoid's own derivative, `O·(1−O)`.
- Binary Cross-Entropy paired with a sigmoid output produces a remarkably clean gradient, `ŷ − y` — this is *why* the two are used together rather than an accident of this particular example.
- Feeding raw, unscaled features into sigmoid-activated layers can saturate the activation and cause hidden-layer gradients to vanish almost entirely — always scale inputs before training a sigmoid-based network.

---

## 11. What's Next

Part 3 shifts from *how* the algorithm is implemented to *why* it works at all — treating the loss as a function of every trainable parameter simultaneously, explaining gradients as the multivariable generalization of a derivative, and covering the update rule, learning rate trade-offs, and what convergence actually means. Look out for the Part 3 notes covering that material next.

---

*Diagrams in this document were generated programmatically to illustrate the concepts discussed above, and are hosted in this repo's [`images/`](https://github.com/versatilerajan/deepcontent/tree/main/images) folder. See also this repo's [`backprop-part1.md`](backprop-part1.md) (the linear-network chain-rule foundations this document builds on) and [`loss-functions.md`](loss-functions.md) (MSE and Binary Cross-Entropy in more depth).*
