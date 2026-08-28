# Backpropagation, Part 1: The Training Loop and the Chain Rule

*Companion notes on backpropagation — the algorithm that trains a neural network by finding the weight and bias values that make its predictions as accurate as possible. Part 1 covers the training cycle, the chain-rule derivation, and a fully worked numeric example on a small CGPA + Skills Score → Placement Score network.*

---

## Table of Contents

1. [What Is Backpropagation?](#1-what-is-backpropagation)
2. [The Training Cycle](#2-the-training-cycle)
3. [The Network for This Example](#3-the-network-for-this-example)
4. [Forward Propagation](#4-forward-propagation)
5. [The Loss Function](#5-the-loss-function)
6. [The Backward Pass: Deriving Every Gradient](#6-the-backward-pass-deriving-every-gradient)
7. [Teacher's Notes](#7-teachers-notes)
8. [Worked Example: 3 Students](#8-worked-example-3-students)
9. [A Bonus Observation: Symmetry](#9-a-bonus-observation-symmetry)
10. [Code: Reproducing This Exact Example](#10-code-reproducing-this-exact-example)
11. [Key Takeaways](#11-key-takeaways)
12. [What's Next](#12-whats-next)

---

## 1. What Is Backpropagation?

**Backpropagation** is the algorithm used to train a neural network. "Training" means searching for the specific values of every weight and bias that make the network's predictions as close as possible to the actual, known outputs in a training dataset.

A network starts out with essentially arbitrary weights and biases, so its first predictions are usually wildly wrong. Backpropagation is the mechanism for taking that wrongness (the **loss**) and turning it into a precise, per-parameter instruction — "increase this weight a little, decrease that one a lot" — using **calculus** (specifically the **chain rule**) to figure out exactly how much each individual weight and bias contributed to the final error.

---

## 2. The Training Cycle

The overall loop backpropagation runs inside looks like this:

![The backpropagation training cycle](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/backprop_cycle.png)

1. **Initialize** — start with some initial values for every weight and bias (this example uses the simplest possible choice: all weights = 1, all biases = 0).
2. **Forward Propagation** — feed one training example's inputs through the network to produce a prediction `ŷ`.
3. **Calculate Loss** — compare `ŷ` to the actual value `y` using a loss function (here, **Mean Squared Error**).
4. **Backward Pass** — use the chain rule to compute how much each individual weight and bias contributed to that loss.
5. **Update Parameters** — nudge every weight and bias in the direction that reduces the loss, using **gradient descent**.

This repeats for every row in the dataset, for as many epochs as you choose to train.

---

## 3. The Network for This Example

To make the chain rule concrete, we'll use a small network predicting a student's **Placement Score** from two features — **CGPA** and **Skills Score**. The network has 2 inputs, one hidden layer of 2 neurons, and 1 output neuron, with **linear** (identity) activations throughout — no sigmoid or ReLU yet, which keeps the calculus as simple as possible for this first pass through backpropagation.

![Network architecture: 2 inputs, 2 hidden neurons, 1 output](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/network_architecture.png)

This uses the exact same notation introduced in the [MLP notes](mlp.md): `w_jk^(l)` connects source node `j` to destination node `k` in target layer `l`; `O_k^(l)` and `b_k^(l)` are indexed by (layer, node).

---

## 4. Forward Propagation

Given one student's features `(x_i1, x_i2)` = (CGPA, Skills Score), the forward pass computes:

```
O11 = w11¹·x_i1 + w21¹·x_i2 + b11
O12 = w12¹·x_i1 + w22¹·x_i2 + b12

ŷ = O21 = w11²·O11 + w21²·O12 + b21
```

Every weight starts at `1` and every bias starts at `0`, per the video's initialization choice.

---

## 5. The Loss Function

This example uses **Mean Squared Error**, matching a single example's squared error:

```
L = (y − ŷ)²
```

(See the [loss functions notes](loss-functions.md) for the full comparison of MSE against other loss functions.)

---

## 6. The Backward Pass: Deriving Every Gradient

The whole point of the chain rule here is to answer: *"If I nudge `w11¹` by a tiny amount, how much does the loss `L` change?"* — for every single weight and bias in the network. Since each parameter influences `L` only through a chain of intermediate computations (`w → O → ŷ → L`), the chain rule multiplies the local derivative at each step of that chain.

![Computational graph: forward pass and backward chain-rule pass](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/computational_graph.png)

**Step 1 — the loss's sensitivity to the prediction:**

```
∂L/∂ŷ = −2(y − ŷ)
```

**Step 2 — output layer parameters** (`ŷ` depends on them directly):

```
∂L/∂w11² = ∂L/∂ŷ · O11
∂L/∂w21² = ∂L/∂ŷ · O12
∂L/∂b21  = ∂L/∂ŷ
```

**Step 3 — hidden layer parameters** (they only affect `L` *through* `ŷ`, so the chain extends one step further, picking up the relevant output-layer weight along the way):

```
∂L/∂w11¹ = ∂L/∂ŷ · w11² · x_i1
∂L/∂w21¹ = ∂L/∂ŷ · w11² · x_i2
∂L/∂b11  = ∂L/∂ŷ · w11²

∂L/∂w12¹ = ∂L/∂ŷ · w21² · x_i1
∂L/∂w22¹ = ∂L/∂ŷ · w21² · x_i2
∂L/∂b12  = ∂L/∂ŷ · w21²
```

Notice the pattern: **every hidden-layer gradient reuses `∂L/∂ŷ`** (already computed in Step 1) and simply multiplies in whichever output-layer weight and input value that particular parameter feeds through. This reuse of already-computed derivatives, rather than recomputing everything from scratch for every parameter, is *exactly* what makes backpropagation efficient — and is the whole reason it has "back" in its name: derivatives computed at the output flow backward and get reused by every earlier layer.

**Step 4 — the update rule** (gradient descent), applied to every weight and bias:

```
w ← w − η · ∂L/∂w
b ← b − η · ∂L/∂b
```

where `η` (eta) is the learning rate.

---

## 7. Teacher's Notes

For reference, here are the original handwritten notes these formulas were transcribed from — the pseudocode for the training loop, and the exact chain-rule derivatives derived on the board:

![Teacher's handwritten notes on the backpropagation algorithm and chain rule](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/teacher_notes_backprop.png)

---

## 8. Worked Example: 3 Students

**Dataset:**

| Student | CGPA (x_i1) | Skills Score (x_i2) | Actual Placement Score (y) |
|---|---|---|---|
| 1 | 8 | 8 | 4 |
| 2 | 7 | 9 | 5 |
| 3 | 6 | 10 | 6 |

**Initial parameters:** all weights = 1, all biases = 0. Learning rate `η = 0.001`.

We process the students one at a time (select a row → predict → compute loss → update), exactly following the training loop in Section 2.

### Student 1 — the "wrong prediction" in full detail

**Forward pass:**
```
O11 = 1×8 + 1×8 + 0 = 16
O12 = 1×8 + 1×8 + 0 = 16
ŷ   = 1×16 + 1×16 + 0 = 32
```

The actual placement score is `4`, but the untrained network predicts `32` — wildly wrong, exactly as expected before any training has happened.

```
error = y − ŷ = 4 − 32 = −28
loss  = (−28)² = 784
```

**Backward pass (chain rule):**
```
∂L/∂ŷ = −2×(−28) = 56

∂L/∂w11² = 56 × O11 = 56 × 16 = 896      ∂L/∂w21² = 56 × O12 = 896      ∂L/∂b21 = 56

∂L/∂w11¹ = 56 × w11² × x_i1 = 56×1×8 = 448    ∂L/∂w21¹ = 56×1×8 = 448    ∂L/∂b11 = 56×1 = 56
∂L/∂w12¹ = 56 × w21² × x_i1 = 56×1×8 = 448    ∂L/∂w22¹ = 56×1×8 = 448    ∂L/∂b12 = 56×1 = 56
```

**Update (η = 0.001):**
```
w11¹ = 1 − 0.001×448 = 0.552        w21¹ = 0.552        b11 = 0 − 0.001×56 = −0.056
w12¹ = 0.552                        w22¹ = 0.552        b12 = −0.056
w11² = 1 − 0.001×896 = 0.104        w21² = 0.104        b21 = −0.056
```

One single update already dragged every weight down substantially — a direct consequence of just how wrong (`error = −28`) the very first prediction was.

### Student 2 and Student 3 — carried through the same process

| Student | ŷ (before update) | error | loss |
|---|---|---|---|
| 2 (CGPA=7, Skills=9, y=5) | 1.769 | 3.231 | 10.437 |
| 3 (CGPA=6, Skills=10, y=6) | 2.800 | 3.200 | 10.241 |

![Loss after each sequential update in epoch 1](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/loss_per_update.png)

*Loss collapses from 784 → 10.4 → 10.2 within a single epoch — almost the entire improvement happens on the very first update, since that's where the prediction was catastrophically wrong. The remaining two updates make much smaller corrections since the predictions are already in a far more reasonable range.*

**Parameters after all 3 students (end of epoch 1):**

```
w11¹ = w12¹ ≈ 0.5629      w21¹ = w22¹ ≈ 0.5683      b11 = b12 ≈ −0.0543
w11² = w21² ≈ 0.2174      b21 ≈ −0.0431
```

---

## 9. A Bonus Observation: Symmetry

Look closely at the updated parameters above: `w11¹` and `w12¹` stayed **identical** throughout training, and so did `w21¹`/`w22¹`, and `w11²`/`w21²`. This isn't a coincidence — since both hidden neurons started with the *exact same* incoming weights, they receive the exact same gradient at every step and therefore stay identical forever. Effectively, the network is only ever learning **one** unique hidden neuron's worth of behavior, duplicated twice.

This is exactly why real implementations **randomly initialize** weights instead of setting them all to the same value — using `weights=1` here was a deliberate teaching simplification to keep the arithmetic easy to follow by hand, but in practice it would waste half the hidden layer's capacity.

---

## 10. Code: Reproducing This Exact Example

```python
import numpy as np

def initialize_parameters():
    # weights=1, biases=0, matching the video's initialization
    return {
        "w1": np.ones((2, 2)),   # input -> hidden weights
        "b1": np.zeros(2),       # hidden biases: [b11, b12]
        "w2": np.ones((2, 1)),   # hidden -> output weights
        "b2": np.zeros(1),       # output bias: [b21]
    }

def forward_propagation(x, params):
    O1 = x @ params["w1"] + params["b1"]     # [O11, O12], linear activation
    yhat = O1 @ params["w2"] + params["b2"]  # scalar prediction
    return O1, yhat[0]

def backward_and_update(x, y, O1, yhat, params, lr):
    error = y - yhat
    dL_dyhat = -2 * error

    # --- output layer gradients ---
    dL_dw2 = dL_dyhat * O1.reshape(-1, 1)      # dL/dw11^2, dL/dw21^2
    dL_db2 = np.array([dL_dyhat])

    # --- hidden layer gradients (chain rule through w2) ---
    dL_dw1 = np.outer(x, dL_dyhat * params["w2"].flatten())
    dL_db1 = dL_dyhat * params["w2"].flatten()

    # --- gradient descent update ---
    params["w2"] -= lr * dL_dw2
    params["b2"] -= lr * dL_db2
    params["w1"] -= lr * dL_dw1
    params["b1"] -= lr * dL_db1

    return error ** 2  # loss for this example

# ---------------------------------------------------------
# Reproduce the worked example above
# ---------------------------------------------------------
params = initialize_parameters()
lr = 0.001
students = [
    (np.array([8, 8]), 4),   # CGPA=8, Skills=8, y=4
    (np.array([7, 9]), 5),
    (np.array([6, 10]), 6),
]

for i, (x, y) in enumerate(students, 1):
    O1, yhat = forward_propagation(x, params)
    loss = backward_and_update(x, y, O1, yhat, params, lr)
    print(f"Student {i}: yhat={yhat:.4f}, loss={loss:.4f}")

print("\nFinal parameters after epoch 1:")
print("w1 (input->hidden):\n", params["w1"])
print("b1 (hidden biases):", params["b1"])
print("w2 (hidden->output):", params["w2"].flatten())
print("b2 (output bias):", params["b2"])
```

Running this prints `yhat=32.0000, loss=784.0000` for Student 1, `yhat=1.7694, loss=10.4367` for Student 2, and `yhat=2.7999, loss=10.2410` for Student 3 — matching every number derived by hand above. The final parameter arrays also confirm the symmetry noted in Section 9: both rows of `w1` and both entries of `w2` stay identical to each other throughout.

---

## 11. Key Takeaways

- Backpropagation trains a network by repeating: **forward pass → compute loss → backward pass (chain rule) → gradient descent update**, for every row, every epoch.
- The chain rule lets every parameter's gradient be computed as a product of local derivatives along the path from that parameter to the loss — and crucially, `∂L/∂ŷ` is computed once and **reused** across every parameter's gradient, which is what makes the algorithm efficient.
- Hidden-layer gradients pick up an extra factor — the relevant output-layer weight — precisely because a hidden neuron only affects the loss *through* the output it feeds into.
- A single very-wrong prediction produces a very large gradient and a large parameter update (Student 1's error of −28 dragged every weight down substantially); as predictions improve, updates naturally shrink.
- Identical weight initialization causes hidden neurons to update identically forever — a strong practical argument for random initialization.

---

## 12. What's Next

This is Part 1 of a three-part series. Having derived the chain rule by hand for a small linear network, natural next steps (Parts 2–3) typically extend this to:

- Networks with **more than one hidden layer**, where the chain rule has to be applied recursively, layer by layer.
- **Non-linear activations** (sigmoid, ReLU) inside the hidden layers, which add an extra activation-derivative factor at every layer.
- **Vectorizing** the entire process across a full batch of examples at once, rather than looping row-by-row as this example did.

---

*Diagrams in this document were generated programmatically to illustrate the concepts discussed above, and are hosted in this repo's [`images/`](https://github.com/versatilerajan/deepcontent/tree/main/images) folder. See also this repo's [`mlp.md`](mlp.md) for the notation this document builds on, and [`loss-functions.md`](loss-functions.md) for more on MSE and other loss functions.*
