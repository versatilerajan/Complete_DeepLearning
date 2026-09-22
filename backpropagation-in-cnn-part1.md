# Backpropagation in CNNs — Part 1: Setting Up the Chain and Deriving the Output Layer

> Part 1 of a 3-part series. This part builds a minimal CNN by hand, writes out every forward
> equation, and derives the backward-pass gradients for the fully-connected (output) layer
> ($\partial L/\partial W_2$, $\partial L/\partial b_2$). Part 2 will continue the chain rule
> back through the flatten, max-pool, ReLU and convolution steps to get $\partial L/\partial W_1$
> and $\partial L/\partial b_1$. Part 3 will assemble everything into gradient descent updates
> and a training loop.

Related notes in this series: [cnn-intro.md](cnn-intro.md) (why CNNs exist, biological
motivation, LeNet/AlexNet timeline) and [ann-vs-cnn.md](ann-vs-cnn.md) (shared dot-product
primitive, weight sharing vs. local connectivity) and [padding-and-strides.md](padding-and-strides.md)
(how output spatial size is controlled). This note assumes familiarity with those and focuses
purely on **how gradients flow backward** through a CNN.

## Table of Contents

1. [Why Backprop in a CNN Deserves Its Own Note](#1-why-backprop-in-a-cnn-deserves-its-own-note)
2. [The Toy Architecture](#2-the-toy-architecture)
3. [Trainable Parameters](#3-trainable-parameters)
4. [Forward Propagation Equations](#4-forward-propagation-equations)
5. [The Backward-Pass Goal](#5-the-backward-pass-goal)
6. [Deriving $\partial L/\partial A_2$: the BCE Loss Derivative](#6-deriving-partial-lpartial-a_2-the-bce-loss-derivative)
7. [Deriving $\partial A_2/\partial Z_2$: the Sigmoid Derivative](#7-deriving-partial-a_2partial-z_2-the-sigmoid-derivative)
8. [Deriving $\partial L/\partial W_2$ and $\partial L/\partial b_2$](#8-deriving-partial-lpartial-w_2-and-partial-lpartial-b_2)
9. [Worked Numerical Example](#9-worked-numerical-example)
10. [From-Scratch NumPy Implementation](#10-from-scratch-numpy-implementation)
11. [Verifying Against a Framework (Autodiff Check)](#11-verifying-against-a-framework-autodiff-check)
12. [Setting Up Part 2: the Rest of the Chain](#12-setting-up-part-2-the-rest-of-the-chain)
13. [Key Takeaways](#13-key-takeaways)
14. [Further Reading](#14-further-reading)

---

## 1. Why Backprop in a CNN Deserves Its Own Note

Every framework hides backprop behind `.backward()` or `GradientTape`. That's great for
building things fast, but it also means the actual mechanism — how a single scalar loss at
the very end of the network turns into 15 separate gradient numbers, one per trainable
parameter, spread across a convolution filter and a dense layer — stays a black box.

This note (and the two that follow) works through that mechanism explicitly, on a network
small enough to write out every number by hand: a 6x6 grayscale image, one 3x3 convolution
filter, ReLU, one max-pool, and a single output neuron. Small enough to verify by hand,
non-trivial enough to contain every mechanism (convolution weight sharing, ReLU's dead-zone,
max-pool's gradient routing) that shows up in real CNNs.

## 2. The Toy Architecture

![Architecture diagram](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01-architecture.png)
*Figure 1 — The forward pass used throughout this series: a 6x6 input is convolved with a
single learned 3x3 filter, passed through ReLU, max-pooled 2x2, flattened, and fed into a
single output neuron with a sigmoid activation, ending in a binary cross-entropy loss. This
is deliberately minimal — one filter, one output neuron — so every gradient can be written
out and checked numerically.*

Concretely:

- **Input**: $X \in \mathbb{R}^{6\times6}$
- **Convolution**: 1 filter, $W_1 \in \mathbb{R}^{3\times3}$, bias $b_1 \in \mathbb{R}$, valid
  padding, stride 1 → output $Z_1 \in \mathbb{R}^{4\times4}$ (since $6-3+1=4$; see
  [padding-and-strides.md](padding-and-strides.md) for why the output shrinks)
- **ReLU**: $A_1 = \text{relu}(Z_1) \in \mathbb{R}^{4\times4}$
- **Max pool**: 2x2 window, stride 2 → $P_1 \in \mathbb{R}^{2\times2}$
- **Flatten**: $F = \text{flatten}(P_1) \in \mathbb{R}^{4\times1}$
- **Output neuron**: $W_2 \in \mathbb{R}^{1\times4}$, $b_2 \in \mathbb{R}$, sigmoid activation
  → $A_2 \in \mathbb{R}$
- **Loss**: binary cross-entropy between $A_2$ and the true label $y$

## 3. Trainable Parameters

| Parameter | Shape | Count |
|---|---|---|
| $W_1$ | $(3,3)$ | 9 |
| $b_1$ | $(1,1)$ | 1 |
| $W_2$ | $(1,4)$ | 4 |
| $b_2$ | $(1,1)$ | 1 |
| **Total** | | **15** |

Every one of these 15 numbers needs its own gradient before a single step of gradient
descent can be taken. That's the entire point of backpropagation: turn one scalar loss into
15 partial derivatives efficiently, by reusing intermediate results via the chain rule
instead of recomputing the forward pass 15 times (once per parameter, as naive finite
differences would require).

## 4. Forward Propagation Equations

![Forward computational graph](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04-forward-graph.png)
*Figure 2 — The forward pass as a computational graph. Each node is an intermediate tensor;
each edge is one equation. $W_1,b_1$ feed into the conv node, $W_2,b_2$ feed into the $Z_2$
node — these four are exactly the leaves that need gradients.*

$$
\begin{aligned}
Z_1 &= \text{conv}(X, W_1) + b_1 \\
A_1 &= \text{relu}(Z_1) \\
P_1 &= \text{maxpool}(A_1) \\
F &= \text{flatten}(P_1) \\
Z_2 &= W_2 F + b_2 \\
A_2 &= \sigma(Z_2) = \frac{1}{1+e^{-Z_2}} \\
L &= -y\log(A_2) - (1-y)\log(1-A_2)
\end{aligned}
$$

**Convolution**, spelled out per output element, for output position $(i,j)$:

$$Z_1[i,j] = \sum_{u=0}^{2}\sum_{v=0}^{2} X[i+u,\,j+v]\cdot W_1[u,v] \;+\; b_1$$

![Convolution sliding window](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02-convolution.png)
*Figure 3 — The 3x3 filter slides across the 6x6 input. At each position it computes an
elementwise product with the underlying patch, sums it, and adds the bias — one scalar
output per position, giving the 4x4 feature map $Z_1$. The same 9 filter weights are reused
at all 16 positions (weight sharing), which is exactly why the convolution gradient later
sums contributions from every position.*

**Max pool** with 2x2 non-overlapping windows keeps only the maximum of each window and
**records which position it came from** — that position record (a mask) is the only thing
that lets gradient flow backward through this layer, since max is not differentiable
everywhere but is differentiable almost everywhere with a well-defined subgradient at the
argmax.

![Maxpool routing](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03-maxpool-routing.png)
*Figure 4 — Left: the 4x4 post-ReLU map $A_1$, with the cell selected by each 2x2 pooling
window highlighted in green. Right: the resulting 2x2 pooled output $P_1$. Only the
highlighted (argmax) cells will receive any gradient during backprop; the other three cells
in each window get exactly zero, because an infinitesimal change to them does not change the
max.*

**Sigmoid and BCE loss** — the two functions that define the tail end of the network:

![Sigmoid and BCE loss curves](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07-sigmoid-bce.png)
*Figure 5 — Left: the sigmoid squashing function that turns $Z_2$ into a probability $A_2 \in
(0,1)$. Right: binary cross-entropy loss as a function of the predicted probability, for both
possible true labels. Notice how steep the $y{=}1$ curve gets as $A_2\to0$ — a confidently
wrong prediction is penalized heavily, which is exactly the property that makes BCE useful for
classification.*

## 5. The Backward-Pass Goal

Backpropagation's job is to compute these four gradients using the chain rule:

$$\frac{\partial L}{\partial W_2},\quad \frac{\partial L}{\partial b_2},\quad
\frac{\partial L}{\partial W_1},\quad \frac{\partial L}{\partial b_1}$$

This note derives the first two (the output/FC layer). The convolution-layer gradients
($\partial L/\partial W_1$, $\partial L/\partial b_1$) require propagating back through
flatten → maxpool → ReLU → conv first, which is the subject of Part 2 — but the full chain is
previewed in [Section 12](#12-setting-up-part-2-the-rest-of-the-chain) so the notation is
consistent across the series.

## 6. Deriving $\partial L/\partial A_2$: the BCE Loss Derivative

Start from the loss and differentiate directly with respect to the prediction $A_2$ (writing
$a$ for $A_2$ and $y_i$ for the true label to keep the algebra compact):

$$\frac{\partial L}{\partial a} = \frac{\partial}{\partial a}\Big[-y_i\log(a) - (1-y_i)\log(1-a)\Big]$$

Differentiate term by term:

$$= -\frac{y_i}{a} + \frac{1-y_i}{1-a} = \frac{-y_i(1-a) + a(1-y_i)}{a(1-a)}$$

Expand the numerator:

$$= \frac{-y_i + y_ia + a - ay_i}{a(1-a)} = \frac{a - y_i}{a(1-a)}$$

$$\boxed{\frac{\partial L}{\partial A_2} = \frac{A_2 - y_i}{A_2(1-A_2)}}$$

Note the $A_2(1-A_2)$ in the denominator — this is not a coincidence. It is exactly the
sigmoid derivative that appears in the next section, and the two will cancel cleanly.

## 7. Deriving $\partial A_2/\partial Z_2$: the Sigmoid Derivative

For $\sigma(z) = \frac{1}{1+e^{-z}}$, the standard derivative is:

$$\frac{\partial A_2}{\partial Z_2} = \sigma(Z_2)\big[1-\sigma(Z_2)\big] = A_2(1-A_2)$$

## 8. Deriving $\partial L/\partial W_2$ and $\partial L/\partial b_2$

The remaining two pieces of the chain are simple, since $Z_2 = W_2F + b_2$ is linear in both:

$$\frac{\partial Z_2}{\partial W_2} = F \qquad \frac{\partial Z_2}{\partial b_2} = 1$$

![Chain rule through FC layer](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05-chain-rule-fc.png)
*Figure 6 — The four-node backward chain for the output layer. Each arrow is one local
derivative; multiplying all of them along the chain (right to left) gives
$\partial L/\partial W_2$ and $\partial L/\partial b_2$.*

Now chain everything together. The key simplification is that $\partial L/\partial A_2$ and
$\partial A_2/\partial Z_2$ share the exact same denominator $A_2(1-A_2)$, so they cancel:

$$
\frac{\partial L}{\partial Z_2} = \frac{\partial L}{\partial A_2}\cdot\frac{\partial A_2}{\partial Z_2}
= \frac{A_2-y_i}{A_2(1-A_2)}\times A_2(1-A_2) = A_2 - y_i
$$

This is the single most useful identity in this derivation: **for sigmoid + binary
cross-entropy, the gradient flowing into the pre-activation $Z_2$ is simply "prediction minus
label."** It is why this specific pairing (sigmoid + BCE, or equivalently softmax +
categorical cross-entropy for multi-class problems) is used almost universally for the output
layer of a classifier — the messy derivative algebra collapses into one clean subtraction.

Continuing the chain for the two parameters:

$$
\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial Z_2}\cdot\frac{\partial Z_2}{\partial W_2}
= (A_2-y_i)\,F
$$

Because $W_2$ is a $(1,4)$ row vector and $F$ is a $(4,1)$ column vector, getting the shapes
to line up requires transposing $F$:

$$\boxed{\frac{\partial L}{\partial W_2} = (A_2-y)\,F^{T}} \qquad \text{shape } (1,4)$$

$$
\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial Z_2}\cdot\frac{\partial Z_2}{\partial b_2}
= (A_2-y_i)\times 1
$$

$$\boxed{\frac{\partial L}{\partial b_2} = A_2 - y} \qquad \text{scalar}$$

A shape check worth doing explicitly, since it is the kind of mistake that is easy to make
and silently wrong in code: $F$ is naturally a $(4,1)$ column vector after flattening $P_1$.
$(A_2-y)$ is a scalar. So $(A_2-y)F$ would be $(4,1)$ — but $W_2$ is $(1,4)$, so the gradient
must also be $(1,4)$. Transposing $F$ to $(1,4)$ fixes this. This kind of shape bookkeeping is
exactly why the class notes explicitly flag "so transpose" at this step — it is the most
common place to introduce a silent bug when implementing backprop from scratch.

## 9. Worked Numerical Example

Running the whole toy network on a fixed random seed (NumPy seed 42) gives concrete numbers,
so every symbolic formula above can be checked against actual output:

| Quantity | Value |
|---|---|
| $X$ | 6x6 random input (see code) |
| $W_1$ | 3x3 random filter |
| $b_1$ | $-0.0720$ |
| $Z_1$ (conv output) | 4x4, e.g. $Z_1[2,0]=3.354$ |
| $A_1$ (post-ReLU) | same shape, negatives zeroed |
| $P_1$ (post-maxpool) | $\begin{bmatrix}1.633 & 1.402\\3.354 & 1.492\end{bmatrix}$ |
| $F$ (flattened) | $[1.633,\ 1.402,\ 3.354,\ 1.492]$ |
| $W_2$ | $[-0.230,\ 0.529,\ 0.172,\ -0.882]$ |
| $b_2$ | $0.0324$ |
| $Z_2$ | $-0.3419$ |
| $A_2$ | $0.4153$ |
| true label $y$ | $1$ |
| $L$ | $0.8786$ |

Applying the boxed formulas from Section 8:

$$\frac{\partial L}{\partial b_2} = A_2 - y = 0.4153 - 1 = -0.5847$$

$$\frac{\partial L}{\partial W_2} = (A_2-y)F^T = -0.5847 \times [1.633,\ 1.402,\ 3.354,\ 1.492]$$
$$= [-0.9548,\ -0.8196,\ -1.9609,\ -0.8724]$$

Both match the from-scratch code output exactly (Section 10) and the independent
autodiff check (Section 11) to within floating-point precision.

## 10. From-Scratch NumPy Implementation

The full forward pass, the hand-derived backward pass, and a central-difference numerical
gradient check (perturb each parameter by $\pm\epsilon$ and measure the loss change directly,
with no calculus at all) were implemented and run:

```python
import numpy as np

def conv2d(X, W, b):
    kh, kw = W.shape
    H, Wd = X.shape
    oh, ow = H - kh + 1, Wd - kw + 1
    Z = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            Z[i, j] = np.sum(X[i:i+kh, j:j+kw] * W) + b
    return Z

def relu(Z):
    return np.maximum(0, Z)

def maxpool2x2(A):
    oh, ow = A.shape[0] // 2, A.shape[1] // 2
    P = np.zeros((oh, ow))
    mask = np.zeros_like(A)          # records the argmax location per window
    for i in range(oh):
        for j in range(ow):
            window = A[2*i:2*i+2, 2*j:2*j+2]
            P[i, j] = np.max(window)
            idx = np.unravel_index(np.argmax(window), window.shape)
            mask[2*i + idx[0], 2*j + idx[1]] = 1
    return P, mask

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, W1, b1, W2, b2, y):
    Z1 = conv2d(X, W1, b1)
    A1 = relu(Z1)
    P1, pool_mask = maxpool2x2(A1)
    F = P1.flatten().reshape(-1, 1)
    Z2 = (W2 @ F + b2).item()
    A2 = sigmoid(Z2)
    L = -(y * np.log(A2) + (1 - y) * np.log(1 - A2))
    return L, dict(Z1=Z1, A1=A1, P1=P1, pool_mask=pool_mask, F=F, Z2=Z2, A2=A2)

def backward_fc_layer(y, cache):
    """Just the piece derived in this note: dL/dW2, dL/db2."""
    A2, F = cache['A2'], cache['F']
    dZ2 = A2 - y            # the "prediction minus label" shortcut from Section 8
    dW2 = dZ2 * F.T         # (1,4)
    db2 = dZ2               # scalar
    return dW2, db2
```

Running a central-difference numerical check against this analytical result:

```python
def numerical_grad(loss_fn, param, eps=1e-5):
    grad = np.zeros_like(param, dtype=float)
    it = np.nditer(param, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        orig = param[idx]
        param[idx] = orig + eps; Lp = loss_fn()
        param[idx] = orig - eps; Lm = loss_fn()
        param[idx] = orig
        grad[idx] = (Lp - Lm) / (2 * eps)
    return grad
```

![Gradient check](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/08-gradient-check.png)
*Figure 7 — Maximum absolute difference between the hand-derived analytical gradient and the
independent numerical (finite-difference) gradient, for every parameter in the toy network,
plotted on a log scale. All four differences are on the order of $10^{-11}$ to $10^{-12}$ —
essentially floating-point noise — confirming the derivation in Section 8 (and the Part 2
derivation for $W_1,b_1$, previewed here) is correct.*

The exact numbers from the run:

```
dW2 max |analytical − numerical| = 3.16e-11
db2 max |analytical − numerical| = 4.37e-12
dW1 max |analytical − numerical| = 1.61e-11   (derived fully in Part 2)
db1 max |analytical − numerical| = 3.79e-12   (derived fully in Part 2)
```

## 11. Verifying Against a Framework (Autodiff Check)

As a second, independent check, the same forward computation was re-implemented with the
`autograd` package (a lightweight reverse-mode automatic-differentiation library) and its
`grad()` function was used to compute all four gradients without any hand-derived formulas at
all:

```python
import autograd.numpy as anp
from autograd import grad

def loss_fn(params):
    W1, b1, W2, b2 = params
    Z1 = anp.stack([anp.stack([anp.sum(X[i:i+3, j:j+3] * W1) + b1
                                for j in range(4)]) for i in range(4)])
    A1 = anp.maximum(0, Z1)
    P1 = anp.stack([anp.stack([anp.max(A1[2*i:2*i+2, 2*j:2*j+2])
                                for j in range(2)]) for i in range(2)])
    F = P1.flatten()
    Z2 = anp.dot(W2.flatten(), F) + b2
    A2 = 1 / (1 + anp.exp(-Z2))
    return -(y * anp.log(A2) + (1 - y) * anp.log(1 - A2))

dW1, db1, dW2, db2 = grad(loss_fn)((W1, b1, W2, b2))
```

Result: the framework's autodiff gradients matched the from-scratch NumPy gradients to
within $5\times10^{-12}$ on every parameter — the same order of agreement as the numerical
check above. Two independent methods (finite differences and reverse-mode autodiff) both
confirm the hand-derived chain rule in Section 8.

## 12. Setting Up Part 2: the Rest of the Chain

$\partial L/\partial W_2$ and $\partial L/\partial b_2$ only required differentiating through
one linear layer. Getting to $\partial L/\partial W_1$ and $\partial L/\partial b_1$ means
continuing the same chain rule back through flatten, max-pool, and ReLU before reaching the
convolution:

![Full backward chain](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06-full-backward-chain.png)
*Figure 8 — The complete backward chain for this network. Part 1 (this note) derived the
right-most two arrows. Part 2 derives the remaining segment: how a gradient with respect to
$F$ becomes a gradient with respect to $P_1$ (a reshape — trivial), then $A_1$ (max-pool
routing via the argmax mask from Figure 4 — the gradient is zero everywhere except the
recorded maximum), then $Z_1$ (ReLU's derivative is 1 where $Z_1>0$ and 0 otherwise), and
finally $W_1$ and $b_1$ (the convolution gradient, which turns out to itself be a
correlation between the input $X$ and the upstream gradient $dZ_1$).*

The full chain rule expressions, for reference in Part 2:

$$
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial A_2}\cdot\frac{\partial A_2}{\partial Z_2}\cdot\frac{\partial Z_2}{\partial F}\cdot\frac{\partial F}{\partial P_1}\cdot\frac{\partial P_1}{\partial A_1}\cdot\frac{\partial A_1}{\partial Z_1}\cdot\frac{\partial Z_1}{\partial W_1}
$$

$$
\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial A_2}\cdot\frac{\partial A_2}{\partial Z_2}\cdot\frac{\partial Z_2}{\partial F}\cdot\frac{\partial F}{\partial P_1}\cdot\frac{\partial P_1}{\partial A_1}\cdot\frac{\partial A_1}{\partial Z_1}\cdot\frac{\partial Z_1}{\partial b_1}
$$

Part 3 will then take all four gradients and apply the gradient-descent update rule with
learning rate $\eta$:

$$W_1 \leftarrow W_1 - \eta\frac{\partial L}{\partial W_1}\qquad b_1 \leftarrow b_1 - \eta\frac{\partial L}{\partial b_1}$$
$$W_2 \leftarrow W_2 - \eta\frac{\partial L}{\partial W_2}\qquad b_2 \leftarrow b_2 - \eta\frac{\partial L}{\partial b_2}$$

## 13. Key Takeaways

- A CNN's trainable parameters are just the convolution filters and the dense-layer weights —
  in this toy network, 15 numbers total ($W_1$: 9, $b_1$: 1, $W_2$: 4, $b_2$: 1).
- Backprop computes all their gradients by applying the chain rule backward through the same
  computational graph used in the forward pass, reusing intermediate results instead of
  recomputing the loss once per parameter.
- For the output layer specifically, differentiating BCE loss with respect to the prediction
  and differentiating sigmoid with respect to its input produce matching $A_2(1-A_2)$ terms
  that cancel, collapsing $\partial L/\partial Z_2$ down to the simple expression
  $A_2 - y$ ("prediction minus label"). This is why sigmoid+BCE (and softmax+cross-entropy)
  is the standard output-layer pairing for classifiers.
- From there, $\partial L/\partial W_2 = (A_2-y)F^T$ and $\partial L/\partial b_2 = A_2-y$ —
  two clean closed-form expressions, verified here against both a finite-difference numerical
  gradient and an independent automatic-differentiation framework, matching to within
  $10^{-11}$.
- Getting matrix shapes to align (row vector $W_2$ vs. column vector $F$) requires care with
  transposes — a common, easy-to-miss source of bugs when implementing backprop by hand.
- The harder part of the chain — propagating gradients back through max-pool's non-smooth
  argmax routing, ReLU's dead zone, and the weight-shared convolution — is set up here and
  solved in Part 2.

## 14. Further Reading

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by
  back-propagating errors*. Nature, 323(6088), 533–536.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to
  document recognition*. Proceedings of the IEEE, 86(11), 2278–2324.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 6 (Deep
  Feedforward Networks — backpropagation) and Chapter 9 (Convolutional Networks). MIT Press.
  https://www.deeplearningbook.org/
- Stanford CS231n, *Convolutional Neural Networks for Visual Recognition* — backpropagation
  and CNN gradient notes: https://cs231n.github.io/optimization-2/ and
  https://cs231n.github.io/convolutional-networks/
- `autograd` documentation (used here as the framework cross-check):
  https://github.com/HIPS/autograd
