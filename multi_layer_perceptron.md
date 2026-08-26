# The Multi-Layer Perceptron (MLP): Notation, Intuition, and Code

*Companion notes covering how a single perceptron's limitations motivate the MLP, the standard notation used for weights/biases/outputs, how to count trainable parameters, and how architecture choices control model flexibility.*

---

## Table of Contents

1. [Why One Perceptron Isn't Enough](#1-why-one-perceptron-isnt-enough)
2. [The Intuition: Combining Perceptrons](#2-the-intuition-combining-perceptrons)
3. [Standard MLP Notation](#3-standard-mlp-notation)
4. [Counting Trainable Parameters](#4-counting-trainable-parameters)
5. [The Forward Pass, Formally](#5-the-forward-pass-formally)
6. [Ways to Increase Flexibility](#6-ways-to-increase-flexibility)
7. [Code: A Minimal MLP Solving XOR](#7-code-a-minimal-mlp-solving-xor)
8. [TensorFlow Playground](#8-tensorflow-playground)
9. [Key Takeaways](#9-key-takeaways)
10. [Further Reading](#10-further-reading)

---

## 1. Why One Perceptron Isn't Enough

A single perceptron can only draw one straight line (or hyperplane) through the data — see the [perceptron notes](perceptron.md) for the full formula and geometry. That's fine when the two classes really are linearly separable, but plenty of simple problems aren't. The classic counterexample is **XOR** (exclusive-or):

| x₁ | x₂ | XOR(x₁, x₂) |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

The two classes sit on opposite diagonal corners of the unit square, so **no single straight line can separate them** — any line you draw will always trap at least one point on the wrong side.

![No single straight line can separate XOR](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/xor_limitation.png)

*Three attempts — horizontal, diagonal, anti-diagonal — and every single one misclassifies at least one point. This isn't a matter of finding the "right" line; provably, no line exists that works. This is exactly the limitation that motivated moving beyond a single perceptron.*

---

## 2. The Intuition: Combining Perceptrons

The fix: run **several perceptrons in parallel**, each drawing its own line, and feed their outputs into another perceptron (typically with a **sigmoid** activation) that combines them. Individually, each hidden neuron still only draws a straight line — but the *combination* of several lines can carve out a non-linear region.

For XOR specifically, two hidden neurons are enough:

- Hidden neuron 1 fires when `x₁ + x₂ > 0.5`
- Hidden neuron 2 fires when `x₁ + x₂ < 1.5`
- The output neuron fires only when **both** hidden neurons fire — i.e., only inside the *band* `0.5 < x₁ + x₂ < 1.5`

![Combining two perceptrons solves XOR](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/combining_perceptrons.png)

*Left and middle: each hidden neuron's own (still straight-line) decision boundary. Right: the output neuron combines both, producing a diagonal band that correctly contains (0,1) and (1,0) while excluding (0,0) and (1,1) — a non-linear decision region built entirely out of linear pieces plus a combining nonlinearity.*

This is the general recipe behind every MLP: stack simple linear decision boundaries and pass them through nonlinear activations, and the combination can approximate decision boundaries of essentially arbitrary complexity (this is the same idea as the Universal Approximation Theorem covered in the [approximation theory notes](README.md)).

---

## 3. Standard MLP Notation

Once a network has more than one layer, keeping track of *which* weight connects *which* nodes requires a consistent notation. The convention used almost everywhere (and the one backpropagation derivations assume):

- **Output of a node**: `O_k^(l)` — the output of node `k` in layer `l`.
- **Bias of a node**: `b_k^(l)` — the bias belonging to node `k` in layer `l`.
- **Weight**: `w_jk^(l)` — the weight connecting node `j` in layer `(l−1)` to node `k` in layer `l`. The superscript `l` refers to the **target (destination) layer**, not the source layer.

![Standard MLP notation](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/mlp_notation.png)

*`w₁₁⁽¹⁾` connects input node 1 to hidden node 1 (destination layer = 1). `w₂₁⁽²⁾` connects hidden node 2 to output node 1 (destination layer = 2). Every bias and output is indexed the same simple way: (layer, node).*

**A good exercise** (the instructor's own recommendation): sketch a more complex network by hand — say 4 inputs, two hidden layers of 5 and 3 nodes, and 2 outputs — and write out the notation for every single weight and bias. Doing this once by hand makes backpropagation's index-heavy formulas far less intimidating later.

---

## 4. Counting Trainable Parameters

Every weight and every bias is a parameter that backpropagation has to learn. Counting them is mechanical once you know the layer sizes:

```
weights between layer (l−1) and layer l  =  n_(l−1) × n_l
biases in layer l                        =  n_l
```

where `n_l` is the number of nodes in layer `l`. (The input layer itself has no biases and no incoming weights — it just holds the raw features.)

![Counting trainable parameters](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/parameter_counting.png)

*For a 3 → 4 → 3 → 2 network: weights = (3×4) + (4×3) + (3×2) = 12+12+6 = 30. Biases = 4+3+2 = 9. Total trainable parameters = 39. Every one of these 39 numbers is something gradient descent has to find during training.*

This quickly explains why modern networks have millions or billions of parameters — it's just this same multiplication, repeated across many wide, deep layers.

---

## 5. The Forward Pass, Formally

Using the notation above, the computation at node `k` in layer `l` is:

```
z_k^(l) = Σⱼ w_jk^(l) · O_j^(l-1)  +  b_k^(l)

O_k^(l) = σ(z_k^(l))
```

where the sum runs over every node `j` in the previous layer, and `σ` is the layer's activation function (sigmoid, ReLU, etc. — see the [ReLU/approximation notes](README.md) for why the choice of `σ` matters so much). Stacking this computation layer by layer, from the input layer through to the output layer, is the entire **forward pass** of an MLP.

---

## 6. Ways to Increase Flexibility

An MLP's capacity — how complex a function it can represent — can be increased along several independent axes, each suited to a different situation:

![Four ways to increase model flexibility](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/architecture_flexibility.png)

- **More hidden nodes** — lets a single hidden layer capture finer-grained detail in the decision boundary (more "lines" available to combine).
- **More input nodes** — needed simply when the dataset has more features to feed in.
- **More output nodes** — needed for multi-class classification (one output per class, typically combined with a softmax) rather than a single binary output.

Depth is its own separate lever:

![Shallow vs. deep networks](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/deep_vs_shallow.png)

- **More hidden layers** turns a shallow network into a **deep neural network**. Depth lets the network build hierarchical features — later layers combine the patterns detected by earlier layers into progressively more abstract concepts — which is typically far more parameter-efficient than making a single hidden layer wider and wider to reach the same representational power.

---

## 7. Code: A Minimal MLP Solving XOR

The following trains a tiny MLP (2 inputs → 2 hidden neurons → 1 output, all sigmoid) from scratch with plain NumPy — no framework — to solve the exact XOR problem a single perceptron cannot:

```python
import numpy as np

# ---------------------------------------------------------
# 1. XOR dataset
# ---------------------------------------------------------
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])  # XOR labels

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    return a * (1 - a)  # a is already sigmoid(z)

# ---------------------------------------------------------
# 2. Initialize a 2 -> 2 -> 1 network
# ---------------------------------------------------------
rng = np.random.default_rng(0)
W1 = rng.normal(size=(2, 2))   # input -> hidden weights, shape (n_in, n_hidden)
b1 = np.zeros((1, 2))
W2 = rng.normal(size=(2, 1))   # hidden -> output weights
b2 = np.zeros((1, 1))

lr = 0.5
epochs = 10000

# ---------------------------------------------------------
# 3. Train with forward pass + backpropagation
# ---------------------------------------------------------
for epoch in range(epochs):
    # ---- forward pass ----
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)              # hidden layer output, O^(1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)              # final output, O^(2) = prediction

    # ---- loss (mean squared error) ----
    loss = np.mean((y - a2) ** 2)

    # ---- backward pass (chain rule, layer by layer) ----
    d_a2 = (a2 - y)                          # dLoss/d(output)
    d_z2 = d_a2 * sigmoid_deriv(a2)          # dLoss/dz2
    d_W2 = a1.T @ d_z2
    d_b2 = d_z2.sum(axis=0, keepdims=True)

    d_a1 = d_z2 @ W2.T                       # error propagated back to hidden layer
    d_z1 = d_a1 * sigmoid_deriv(a1)
    d_W1 = X.T @ d_z1
    d_b1 = d_z1.sum(axis=0, keepdims=True)

    # ---- gradient descent update ----
    W2 -= lr * d_W2
    b2 -= lr * d_b2
    W1 -= lr * d_W1
    b1 -= lr * d_b1

    if epoch % 2000 == 0:
        print(f"epoch {epoch:5d}  loss = {loss:.4f}")

print("\nFinal predictions:")
print(np.round(a2, 3))
print("Trainable parameters:", W1.size + b1.size + W2.size + b2.size)  # 2*2 + 2 + 2*1 + 1 = 9
```

Running this should show the loss dropping steadily toward ~0 and the final predictions landing close to `[0, 1, 1, 0]` — the exact XOR pattern a lone perceptron could never learn. Note the parameter count matches the formula from Section 4: `(2×2 + 2) + (2×1 + 1) = 6 + 3 = 9`.

---

## 8. TensorFlow Playground

[TensorFlow Playground](https://playground.tensorflow.org/) is a browser-based, no-code way to build intuition for everything above: you can add/remove hidden layers, add/remove neurons per layer, switch activation functions (sigmoid, tanh, ReLU, linear), and watch the decision boundary reshape in real time on datasets like XOR, spirals, and concentric circles. A few things worth trying there directly:

- Solve the XOR-shaped dataset with just 1 hidden neuron (it can't) vs. 2+ hidden neurons (it can) — a live version of Sections 1–2 above.
- Switch the hidden layer's activation from **sigmoid** to **ReLU** and watch how much faster training converges, and how the decision boundary becomes piecewise-linear (faceted) rather than smoothly curved.
- Try the spiral dataset with a shallow (1 hidden layer) vs. deep (3+ hidden layers) network to see depth's effect on capacity directly.

---

## 9. Key Takeaways

- A single perceptron can only represent a linear decision boundary — problems like XOR are provably unsolvable by one.
- Running several perceptrons in parallel and combining their outputs (typically through a sigmoid) lets the network carve out non-linear decision regions from simple linear pieces.
- The standard notation `w_jk^(l)`, `b_k^(l)`, `O_k^(l)` indexes every weight by (source node, destination node, destination layer), and every bias/output by (layer, node) — worth memorizing before tackling backpropagation.
- Trainable parameters = sum over layers of `(n_(l-1) × n_l) + n_l` — this is exactly what backpropagation has to solve for.
- Model flexibility can be scaled independently along four axes: more hidden nodes, more inputs, more outputs, and more hidden layers (depth) — each suited to a different kind of added complexity in the data.

---

## 10. Further Reading

- Minsky, M. & Papert, S. (1969). *Perceptrons* — the original proof of single-layer limitations (including XOR).
- Rumelhart, D., Hinton, G., & Williams, R. (1986). *Learning Representations by Back-Propagating Errors* — the paper that popularized backpropagation for training MLPs.
- Nielsen, M. *Neural Networks and Deep Learning* (free online book) — a very clear treatment of MLP notation and backpropagation derivations.
- [TensorFlow Playground](https://playground.tensorflow.org/) — interactive architecture exploration referenced in Section 8.
- See also this repo's [`perceptron.md`](perceptron.md) (single-neuron foundations) and [`README.md`](README.md) (approximation theory / why deep networks can represent complex functions at all).

---

*Diagrams in this document were generated programmatically to illustrate the concepts discussed above, and are hosted in this repo's [`images/`](https://github.com/versatilerajan/deepcontent/tree/main/images) folder.*
