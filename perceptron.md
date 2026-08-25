# The Perceptron: Formula, Geometry, Loss Function, and Code

*Companion notes on the Perceptron — from the geometric "perceptron trick" to a proper loss-function-and-gradient-descent formulation, and how the same architecture generalizes into logistic regression, linear regression, and more.*

---

## Table of Contents

1. [What Is a Perceptron?](#1-what-is-a-perceptron)
2. [Labeling Regions: The Decision Boundary](#2-labeling-regions-the-decision-boundary)
3. [How Weights and Bias Transform the Boundary](#3-how-weights-and-bias-transform-the-boundary)
4. [The Perceptron Trick (Algorithm)](#4-the-perceptron-trick-algorithm)
5. [Worked Numerical Example](#5-worked-numerical-example)
6. [From Heuristic to Loss Function](#6-from-heuristic-to-loss-function)
7. [Training via Gradient Descent](#7-training-via-gradient-descent)
8. [Flexibility: One Architecture, Many Models](#8-flexibility-one-architecture-many-models)
9. [Code: Training a Perceptron From Scratch](#9-code-training-a-perceptron-from-scratch)
10. [Key Takeaways](#10-key-takeaways)
11. [Further Reading](#11-further-reading)

---

## 1. What Is a Perceptron?

A **perceptron** is the simplest possible neural network: a single artificial neuron that makes a **binary classification** decision by computing a weighted sum of its inputs, adding a bias, and passing the result through a step function.

**Formula:**

```
z = w·x + b = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

ŷ = step(z) =  { +1   if z ≥ 0
               { −1   if z < 0
```

where `x = (x₁, ..., xₙ)` is the input feature vector, `w = (w₁, ..., wₙ)` are the learned weights, and `b` is the learned bias (also called the intercept).

![A single perceptron: inputs, weights, sum, bias, activation](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/perceptron_architecture.png)

*Each input is multiplied by its own weight, everything (plus the bias) is summed into `z`, and the step function converts `z` into a hard ±1 decision. Geometrically, `w·x + b = 0` defines a straight line (in 2D) or a hyperplane (in higher dimensions) — the perceptron can only correctly classify data that some such line can separate, i.e. data that is **linearly separable**.*

---

## 2. Labeling Regions: The Decision Boundary

The equation `w·x + b = 0` defines the **decision boundary**. Every point on one side of it gets one label; every point on the other side gets the other:

- `w·x + b > 0` → predict `ŷ = +1`
- `w·x + b < 0` → predict `ŷ = −1`
- `w·x + b = 0` → exactly on the boundary

The weight vector `w` has a nice geometric meaning: it is always **perpendicular to the decision boundary**, pointing toward the positive region.

![Decision boundary with labeled positive/negative regions](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/decision_boundary.png)

*The black line is where `w·x + b = 0`. The green region is where the perceptron predicts +1; the red region is where it predicts −1. The purple arrow is the weight vector `w`, sitting perpendicular to the boundary — this is always true for a linear decision boundary.*

---

## 3. How Weights and Bias Transform the Boundary

Two knobs control the line, and they do two very different things:

- **Changing the bias `b`** shifts the line **parallel to itself** (it moves the line without rotating it), since `b` alone controls where the line crosses each axis.
- **Changing a weight `wᵢ`** **rotates** the line, since the weights control the line's slope/orientation.

![Effect of changing bias vs. weight on the decision boundary](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bias_weight_transform.png)

*Left: as bias increases/decreases, the line slides up/down without changing angle. Right: as a weight changes sign and magnitude, the line pivots around the origin. Training a perceptron is really just a search over all possible (w, b) combinations — i.e., all possible shifted-and-rotated lines — for one that separates the classes.*

---

## 4. The Perceptron Trick (Algorithm)

The classic **perceptron trick** is a simple, greedy way to search for a separating line: whenever the current line **misclassifies** a point, nudge the line *toward* that point so it's more likely to be classified correctly next time.

**Update rule** — for a misclassified point `(x, y)` where `y ∈ {+1, −1}`:

```
w ← w + η · y · x
b ← b + η · y
```

where `η` (eta) is the **learning rate**, a small positive number (e.g. 0.01–1.0) that controls how big each nudge is. Using a learning rate instead of jumping straight to a "corrected" line prevents wild, unstable, single-step adjustments — especially important when there's noisy or non-perfectly-separable data.

**Full training loop:**

1. Initialize `w` and `b` (commonly to zeros, or small random values).
2. For a fixed number of **epochs** (full passes over the data):
   - For each training point `(x, y)`:
     - Compute `z = w·x + b`
     - If the point is misclassified (i.e. `y · z ≤ 0`), apply the update rule above.
3. Stop when an epoch produces no misclassifications (converged), or the epoch budget runs out.

**Convergence guarantee:** if the data really is linearly separable, the Perceptron Convergence Theorem guarantees this process finds a separating line in a finite number of steps. If the data is *not* linearly separable, it will never fully converge — it will keep adjusting forever, which is one of the trick's key limitations (addressed in [Section 6](#6-from-heuristic-to-loss-function)).

---

## 5. Worked Numerical Example

Let's trace through the algorithm by hand on a tiny dataset, starting from deliberately bad initial values so we can see multiple updates happen.

**Data** (8 points, 2 classes):

| Point | Label y |
|---|---|
| (2, 3), (3, 4), (4, 3.5), (3.5, 2.5) | +1 |
| (−1, −1), (−2, 0), (0, −2), (−1.5, −2.5) | −1 |

**Initial parameters:** `w = (−3, −3)`, `b = 5`, learning rate `η = 1`.

| Step | Point (x, y_label) | z = w·x + b | y·z | Action | New w | New b |
|---|---|---|---|---|---|---|
| 1 | (2,3), +1 | −3(2)−3(3)+5 = −10 | −10 ≤ 0 | misclassified → update | (−1, 0) | 6 |
| 2 | (3,4), +1 | −1(3)+0(4)+6 = 3 | 3 > 0 | correct, no change | (−1, 0) | 6 |
| 3 | (4,3.5), +1 | −1(4)+0+6 = 2 | 2 > 0 | correct | (−1, 0) | 6 |
| 4 | (3.5,2.5), +1 | −1(3.5)+0+6 = 2.5 | 2.5 > 0 | correct | (−1, 0) | 6 |
| 5 | (−1,−1), −1 | −1(−1)+0(−1)+6 = 7 | −7 ≤ 0 | misclassified → update | (0, 1) | 5 |
| 6 | (−2,0), −1 | 0(−2)+1(0)+5 = 5 | −5 ≤ 0 | misclassified → update | (2, 1) | 4 |
| 7 | (0,−2), −1 | 2(0)+1(−2)+4 = 2 | −2 ≤ 0 | misclassified → update | (2, 3) | 3 |
| 8 | (−1.5,−2.5), −1 | 2(−1.5)+3(−2.5)+3 = −7.5 | 7.5 > 0 | correct | (2, 3) | 3 |

After just **one epoch and 4 updates**, we reach `w = (2, 3)`, `b = 3`. Checking all 8 points again confirms every one is now correctly classified — the algorithm has converged.

![Perceptron trick converging over 4 updates](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/perceptron_convergence.png)

*The dashed gray line is the (bad) starting boundary. The solid orange-to-black progression shows the boundary after each of the 4 updates, visibly rotating and shifting toward a line that cleanly separates the green (+1) and red (−1) points, ending at the solid black final boundary.*

---

## 6. From Heuristic to Loss Function

The perceptron trick works, but it has a real weakness: **it has no way to measure "how good" a candidate line is.** It only asks a yes/no question — "is this point currently misclassified?" — with no sense of *how badly* misclassified, and no principled way to compare two different candidate lines against each other.

A **loss function** fixes this by turning classification quality into a single number we can actually optimize.

**The Perceptron Loss** (used by scikit-learn's `Perceptron`), for one training example `(x, y)`:

```
L(w, b) = max(0, −y · (w·x + b)) = max(0, −y · f(x))
```

where `f(x) = w·x + b`. Notice:

- If the point is **correctly classified** (`y · f(x) > 0`), the loss is exactly `0` — perfectly classified points don't push the boundary around at all.
- If the point is **misclassified** (`y · f(x) < 0`), the loss grows **linearly** with how far past the boundary (on the wrong side) the point is — the more confidently wrong, the bigger the penalty.

This is closely related to (but slightly different from) the **hinge loss** used in Support Vector Machines, `max(0, 1 − y·f(x))`, which additionally penalizes points that are correctly classified but too close to the boundary (inside the "margin").

![The perceptron loss function vs. hinge loss](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/perceptron_loss.png)

*Perceptron loss (red, solid) is exactly zero for any correctly classified point, no matter how close to the boundary. Hinge loss (blue, dashed) still charges a shrinking penalty up until the point is a full margin of 1 away from the boundary — this small difference is what gives SVMs their characteristic "maximum margin" boundary, while a perceptron is happy with any separating line at all.*

The **total loss** over a dataset of `m` points is just the sum (or average) over all of them:

```
L(w, b) = (1/m) · Σᵢ max(0, −yᵢ · (w·xᵢ + b))
```

---

## 7. Training via Gradient Descent

With a loss function in hand, we can train the perceptron the same way we train any other differentiable model: **gradient descent.**

For a single misclassified point, the loss is `L = −y·(w·x + b)`. Taking partial derivatives:

```
∂L/∂w = −y · x
∂L/∂b = −y
```

(For a correctly classified point, both derivatives are `0`, since the loss there is a flat `0`.)

The gradient descent update rule subtracts a step in the direction of the gradient:

```
w ← w − η · ∂L/∂w = w − η·(−y·x) = w + η·y·x
b ← b − η · ∂L/∂b = b − η·(−y)   = b + η·y
```

**This is exactly the perceptron trick from Section 4.** The "trick" wasn't an ad-hoc heuristic after all — it's precisely gradient descent on the perceptron loss function, just discovered/taught in reverse-historical order. Framing it this way is powerful because it immediately connects the perceptron to the entire toolbox of gradient-based optimization: mini-batching, momentum, adaptive learning rates (Adam, etc.), regularization, and so on all become available.

---

## 8. Flexibility: One Architecture, Many Models

The perceptron's `w·x + b` structure — a linear combination of inputs — turns out to be the shared skeleton underneath several other foundational ML models. Swap out the **activation function** and/or the **loss function**, and the same architecture is repurposed for a different task entirely:

| Model | Activation | Loss function | Task |
|---|---|---|---|
| Perceptron | Step function | Perceptron loss | Hard binary classification |
| Logistic Regression | Sigmoid | Binary Cross-Entropy | Probabilistic binary classification |
| Softmax Regression | Softmax | Categorical Cross-Entropy | Multi-class classification |
| Linear Regression | Identity (none) | Mean Squared Error | Continuous value prediction |
| (Soft-margin) SVM | Step function (at inference) | Hinge Loss | Max-margin binary classification |

This is the same underlying lesson from the neural-network / approximation-theory discussion: the *architecture* (`w·x + b`) provides the representational scaffold, while the choice of activation and loss shapes exactly what kind of function you're able to learn and how "wrong answers" get penalized during training.

---

## 9. Code: Training a Perceptron From Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Synthetic linearly-separable dataset
# ---------------------------------------------------------
rng = np.random.default_rng(42)
X_pos = rng.normal(loc=[3, 3], scale=0.8, size=(50, 2))
X_neg = rng.normal(loc=[-1, -1], scale=0.8, size=(50, 2))
X = np.vstack([X_pos, X_neg])
y = np.array([1] * 50 + [-1] * 50)

# ---------------------------------------------------------
# 2. Perceptron trick training loop
# ---------------------------------------------------------
def train_perceptron(X, y, lr=0.1, epochs=20):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    history = []  # track weights/bias for visualizing convergence

    for epoch in range(epochs):
        misclassified = 0
        for xi, yi in zip(X, y):
            z = np.dot(w, xi) + b
            if yi * z <= 0:          # misclassified point
                w += lr * yi * xi     # perceptron update rule
                b += lr * yi
                misclassified += 1
        history.append((w.copy(), b))
        if misclassified == 0:
            print(f"Converged after {epoch + 1} epochs.")
            break
    return w, b, history

w, b, history = train_perceptron(X, y, lr=0.1, epochs=20)
print("Final weights:", w, " bias:", b)

# ---------------------------------------------------------
# 3. Visualize the learned decision boundary
# ---------------------------------------------------------
xx = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
yy = -(w[0] * xx + b) / w[1]

plt.scatter(X_pos[:, 0], X_pos[:, 1], color="green", label="y = +1")
plt.scatter(X_neg[:, 0], X_neg[:, 1], color="red", label="y = -1")
plt.plot(xx, yy, color="black", linewidth=2, label="learned boundary")
plt.legend()
plt.xlabel("x1"); plt.ylabel("x2")
plt.title("Perceptron: Learned Decision Boundary")
plt.show()
```

**Equivalent loss-function / gradient-descent formulation** (mathematically identical to the loop above — this makes the connection from Section 7 explicit in code):

```python
def train_perceptron_gd(X, y, lr=0.1, epochs=20):
    w = np.zeros(X.shape[1])
    b = 0.0
    for epoch in range(epochs):
        total_loss = 0.0
        for xi, yi in zip(X, y):
            margin = yi * (np.dot(w, xi) + b)
            loss = max(0, -margin)
            total_loss += loss
            if margin <= 0:                 # gradient is nonzero only when loss > 0
                grad_w = -yi * xi
                grad_b = -yi
                w -= lr * grad_w             # w += lr * yi * xi
                b -= lr * grad_b             # b += lr * yi
        if total_loss == 0:
            print(f"Converged after {epoch + 1} epochs (loss = 0).")
            break
    return w, b
```

**Using scikit-learn** (production-grade version of the same idea):

```python
from sklearn.linear_model import Perceptron

clf = Perceptron(eta0=0.1, max_iter=1000, tol=1e-3)
clf.fit(X, y)
print("Weights:", clf.coef_, " Bias:", clf.intercept_)
```

---

## 10. Key Takeaways

- A perceptron computes `ŷ = step(w·x + b)` — a linear combination of inputs passed through a hard threshold.
- The decision boundary `w·x + b = 0` is a straight line (or hyperplane); `w` is always perpendicular to it.
- Changing `b` shifts the boundary in parallel; changing `w` rotates it.
- The **perceptron trick** — nudge the line toward misclassified points — converges in finite steps *if and only if* the data is linearly separable.
- The **perceptron loss**, `max(0, −y·f(x))`, gives the trick a rigorous foundation: gradient descent on this loss produces *exactly* the same update rule as the heuristic trick.
- Swapping the activation and loss functions turns the exact same `w·x + b` skeleton into logistic regression, softmax regression, linear regression, or an SVM — the perceptron is really a template, not just one fixed model.

---

## 11. Further Reading

- Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain* — the original paper.
- Minsky, M. & Papert, S. (1969). *Perceptrons* — the famous critique of single-layer perceptrons' limitations (e.g. XOR), which motivated multi-layer networks.
- Novikoff, A. (1962). *On Convergence Proofs for Perceptrons* — the convergence guarantee for linearly separable data.
- Scikit-learn documentation on [`Perceptron`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html) and its relationship to `SGDClassifier` with `loss="perceptron"`.
- Cortes, C. & Vapnik, V. (1995). *Support-Vector Networks* — hinge loss and the max-margin idea contrasted with the perceptron loss.

---

*Diagrams in this document were generated programmatically to illustrate the concepts discussed above, and are hosted in this repo's [`images/`](https://github.com/versatilerajan/deepcontent/tree/main/images) folder.*
