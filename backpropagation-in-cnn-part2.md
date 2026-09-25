# Backpropagation in CNNs — Part 2: Flatten, Max Pool and Convolution

> Part 2 of a 3-part series. [Part 1](backpropagation-in-cnn-part1.md) derived the output-layer
> gradients $\partial L/\partial W_2$ and $\partial L/\partial b_2$ and stopped at the boundary
> of the flatten layer. This part continues the same chain rule backward through flatten, max
> pooling, ReLU, and finally the convolution layer itself, arriving at
> $\partial L/\partial W_1$ and $\partial L/\partial b_1$ — the last two of the network's 15
> trainable parameters. Part 3 will assemble all four gradients into the gradient-descent
> update loop.

Related notes: [backpropagation-in-cnn-part1.md](backpropagation-in-cnn-part1.md) (architecture,
forward pass, FC-layer derivation), [cnn-intro.md](cnn-intro.md), [ann-vs-cnn.md](ann-vs-cnn.md),
[padding-and-strides.md](padding-and-strides.md).

## Table of Contents

1. [Where Part 1 Left Off](#1-where-part-1-left-off)
2. [Flatten Layer Backward](#2-flatten-layer-backward)
3. [Max Pooling Layer Backward](#3-max-pooling-layer-backward)
4. [ReLU Layer Backward](#4-relu-layer-backward)
5. [Convolution Layer Backward — Setting Up](#5-convolution-layer-backward--setting-up)
6. [Deriving $\partial L/\partial b_1$](#6-deriving-partial-lpartial-b_1)
7. [Deriving $\partial L/\partial W_1$](#7-deriving-partial-lpartial-w_1)
8. [Worked Numerical Example, Step by Step](#8-worked-numerical-example-step-by-step)
9. [From-Scratch NumPy Implementation](#9-from-scratch-numpy-implementation)
10. [Verifying Against Part 1's Preview and a Framework](#10-verifying-against-part-1s-preview-and-a-framework)
11. [Setting Up Part 3](#11-setting-up-part-3)
12. [Key Takeaways](#12-key-takeaways)
13. [Further Reading](#13-further-reading)

---

## 1. Where Part 1 Left Off

Part 1 derived, for the output neuron:

$$\frac{\partial L}{\partial Z_2} = A_2 - y \qquad
\frac{\partial L}{\partial W_2} = (A_2-y)F^T \qquad
\frac{\partial L}{\partial b_2} = A_2 - y$$

What Part 1 did **not** derive is how a gradient with respect to $F$ (the flattened vector
feeding into the output neuron) turns into gradients with respect to $W_1$ and $b_1$ — the
convolution filter's weights and bias. That requires stepping backward through three more
layers: flatten, max pool, and ReLU, before finally reaching the convolution itself.

![Full backward chain](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/14-full-chain-part1-part2.png)
*Figure 1 — The complete backward chain across both parts. Part 1 covered the two right-most
arrows ($W_1 \leftarrow \dots \leftarrow F$, wait — read right to left: $Z_2 \to F$ is Part 1's
territory in reverse). This part covers everything from $F$ back to $W_1$.*

The very first link in this part's chain is $\partial Z_2/\partial F$. Since $Z_2 = W_2F+b_2$
is linear in $F$:

$$\frac{\partial Z_2}{\partial F} = W_2$$

so combined with Part 1's $\partial L/\partial Z_2 = A_2-y$:

$$\frac{\partial L}{\partial F} = (A_2-y)\,W_2 \qquad \text{shape } (1,4)$$

This is the starting point for everything below.

## 2. Flatten Layer Backward

Flatten has **no trainable parameters** — it only reshapes a $(2,2)$ tensor into a $(4,1)$
vector (or here, viewed as the $(1,4)$ row above). Because reshaping doesn't change any
values, only their layout in memory, the backward pass is just the reverse reshape: take
whatever gradient arrived with respect to $F$, and reshape it back into $P_1$'s original
$(2,2)$ shape.

$$\frac{\partial F}{\partial P_1} = \text{reshape}(P_1.\text{shape}) \qquad
\frac{\partial L}{\partial P_1} = \text{reshape}\!\left(\frac{\partial L}{\partial F},\ P_1.\text{shape}\right)$$

![Flatten backward](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/09-flatten-backward.png)
*Figure 2 — Flatten's backward pass is literally just un-flattening: the same four numbers,
same order, reshaped from a $(4,1)$ column back into $P_1$'s $(2,2)$ grid. No computation, no
parameters — just a change of shape.*

## 3. Max Pooling Layer Backward

Max pooling also has **no trainable parameters**, but unlike flatten it isn't a pure
reshape — it's a lossy selection (3 of every 4 values in each window are discarded in the
forward pass). This raises the natural question: where does the gradient for those 3
discarded values go?

The answer follows directly from what "gradient" means: $\partial L/\partial A_1[m,n]$ asks
*"if I nudge $A_1[m,n]$ by an infinitesimal amount, how much does $L$ change?"* For any cell
that was **not** the maximum in its window, nudging it slightly doesn't change the max (as
long as the nudge is small enough not to overtake the actual max) — so it doesn't change
$P_1$, and therefore doesn't change $L$ at all. Its gradient is exactly zero. Only the cell
that **was** the max sees its nudge pass straight through to $P_1$ unchanged, so it inherits
the full upstream gradient.

$$
\frac{\partial L}{\partial A_1[m,n]} =
\begin{cases}
\dfrac{\partial L}{\partial P_1[i,j]} & \text{if } A_1[m,n] \text{ was the max of pooling window } (i,j) \\[4pt]
0 & \text{otherwise}
\end{cases}
$$

This is why the forward pass needs to **record**, for every pooling window, which position
won — that record (often called the "pooling mask" or "switches") is what makes this
backward step possible; without it there would be no way to know where to route the gradient.

![Maxpool backward](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/10-maxpool-backward.png)
*Figure 3 — Left: the forward pass, with the argmax cell of each 2x2 window highlighted in
green — this is the mask recorded during the forward pass. Right: the backward pass. Each of
the four incoming gradient values from $\partial L/\partial P_1$ is placed at exactly the cell
that produced it; every other cell — 12 of the 16 in total here — receives precisely 0.*

## 4. ReLU Layer Backward

ReLU's derivative is piecewise constant — 1 where the pre-activation was positive, 0
otherwise (the point $Z_1=0$ is a measure-zero edge case, conventionally taken as 0 or 1; it
never matters in practice):

$$\frac{\partial A_1}{\partial Z_1} =
\begin{cases}
1 & \text{if } Z_1[i,j] > 0 \\
0 & \text{if } Z_1[i,j] < 0
\end{cases}$$

So the backward pass through ReLU is an elementwise gate: multiply the incoming gradient by
this 0/1 mask.

$$\frac{\partial L}{\partial Z_1} = \frac{\partial L}{\partial A_1} \odot \mathbb{1}[Z_1>0]$$

![ReLU backward](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/11-relu-backward.png)
*Figure 4 — Left: ReLU and its derivative, a step function that is exactly 1 for positive
inputs and 0 for negative ones. Right: the elementwise gate applied to this example's 4x4
map — green cells (where $Z_1>0$) pass their incoming gradient through unchanged, red cells
(where $Z_1 \le 0$) get killed to exactly 0, regardless of what gradient arrived.*

Notice that after max pooling already zeroed out 3 of every 4 cells, and ReLU zeroes out any
cell where the original pre-activation was negative, the resulting $\partial L/\partial Z_1$
is **sparse** — in this toy example only 3 of the 16 cells end up nonzero. This sparsity is a
direct, mechanical consequence of max pooling and ReLU, not a special property of this
example.

## 5. Convolution Layer Backward — Setting Up

The convolution's forward pass, for a $3\times3$ filter $W_1$ sliding over a $6\times6$ input
$X$ with valid padding, produces each output element as:

$$Z_1[i,j] = \sum_{u=0}^{2}\sum_{v=0}^{2} X[i+u,\,j+v]\cdot W_1[u,v] + b_1$$

Both $b_1$ and every entry of $W_1$ are reused across **all 16** output positions (weight
sharing — see [ann-vs-cnn.md](ann-vs-cnn.md)). This is the crucial structural difference from
the dense (FC) layer in Part 1: there, each weight touched exactly one output; here, each
weight touches every output position. So by the multivariate chain rule, the gradient for
each shared parameter must be a **sum over all the positions it participated in**:

$$\frac{\partial L}{\partial b_1} = \sum_{i,j}\frac{\partial L}{\partial Z_1[i,j]}\cdot\frac{\partial Z_1[i,j]}{\partial b_1}
\qquad\qquad
\frac{\partial L}{\partial W_1[u,v]} = \sum_{i,j}\frac{\partial L}{\partial Z_1[i,j]}\cdot\frac{\partial Z_1[i,j]}{\partial W_1[u,v]}$$

## 6. Deriving $\partial L/\partial b_1$

Since $b_1$ appears as a plain additive constant in every one of the 16 equations for
$Z_1[i,j]$, its local derivative is trivially 1 at every position:

$$\frac{\partial Z_1[i,j]}{\partial b_1} = 1 \quad \text{for all } (i,j)$$

Substituting into the sum from Section 5, the sum collapses to just adding up every entry of
the upstream gradient matrix:

$$\boxed{\frac{\partial L}{\partial b_1} = \sum_{i,j}\frac{\partial L}{\partial Z_1[i,j]} = \text{sum}\!\left(\frac{\partial L}{\partial Z_1}\right)}$$

![Bias gradient as a sum](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/12-bias-gradient-sum.png)
*Figure 5 — The convolution bias gradient is simply the sum of every element of the 4x4
upstream gradient matrix $\partial L/\partial Z_1$ — a direct consequence of $b_1$ being added
identically at all 16 output positions.*

## 7. Deriving $\partial L/\partial W_1$

For $W_1[u,v]$, differentiate $Z_1[i,j]=\sum_{u,v}X[i+u,j+v]W_1[u,v]+b_1$ with respect to one
specific entry $W_1[u,v]$: every term in the double sum vanishes except the one matching that
$(u,v)$, leaving:

$$\frac{\partial Z_1[i,j]}{\partial W_1[u,v]} = X[i+u,\,j+v]$$

Substituting into the sum from Section 5:

$$\boxed{\frac{\partial L}{\partial W_1[u,v]} = \sum_{i,j} X[i+u,j+v]\cdot\frac{\partial L}{\partial Z_1[i,j]}}$$

This expression has a compact interpretation: it is exactly the **cross-correlation** (a
convolution without kernel-flipping) of the input $X$ with the upstream gradient
$\partial L/\partial Z_1$, using $\partial L/\partial Z_1$ itself as the "filter" being slid
across $X$:

$$\frac{\partial L}{\partial W_1} = \text{conv}\!\left(X,\ \frac{\partial L}{\partial Z_1}\right)$$

which is the "specific deconvolution-like operation" this derivation is often described as —
it looks like a convolution, but it convolves the **original input** with the **gradient**
rather than with a filter, and it produces a gradient the same shape as the original filter
instead of a feature map.

![Convolution weight gradient](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/13-conv-weight-gradient.png)
*Figure 6 — Computing $\partial L/\partial W_1$: the 4x4 upstream gradient
$\partial L/\partial Z_1$ (center, red) is correlated against the 6x6 input $X$ (left) —
conceptually the same sliding-window operation as the forward convolution, but sliding the
gradient over the input — producing the 3x3 gradient for $W_1$ (right, green). The red box on
$X$ marks the sub-patch aligned with the top-left offset of this correlation.*

## 8. Worked Numerical Example, Step by Step

Using the same fixed toy network from Part 1 (NumPy seed 42), the full backward chain
computed step by step:

| Step | Quantity | Result |
|---|---|---|
| Start | $\partial L/\partial Z_2 = A_2-y$ | $-0.5847$ |
| §1 | $\partial L/\partial F = (A_2-y)W_2$ | $[0.1347,\ {-}0.3090,\ {-}0.1004,\ 0.5154]$ |
| §2 | $\partial L/\partial P_1$ (reshaped) | $\begin{bmatrix}0.1347 & -0.3090\\-0.1004 & 0.5154\end{bmatrix}$ |
| §3 | $\partial L/\partial A_1$ (routed) | 4x4, nonzero only at the 4 argmax cells |
| §4 | $\partial L/\partial Z_1$ (ReLU-gated) | 4x4, all 4 argmax cells had $Z_1>0$, so all 4 pass through unchanged (12 of 16 cells are 0 from max-pool, and no additional cell is killed by ReLU here) |
| §6 | $\partial L/\partial b_1 = \text{sum}(\cdot)$ | $0.2406$ |
| §7 | $\partial L/\partial W_1 = \text{conv}(X, \cdot)$ | $\begin{bmatrix}-1.014 & 0.027 & -0.118\\1.233 & 0.442 & -0.273\\-0.436 & -0.096 & 0.014\end{bmatrix}$ |

These are exactly the same $\partial L/\partial W_1$ and $\partial L/\partial b_1$ values that
were previewed (but not derived) at the end of Part 1 — this part supplies the missing
derivation and confirms the numbers agree to floating-point precision.

## 9. From-Scratch NumPy Implementation

Continuing directly from Part 1's cache (`Z1`, `A1`, `P1`, `pool_mask`, `F`, `W2`, `A2`):

```python
import numpy as np

# --- inherited from Part 1's forward pass ---
# dZ2 = A2 - y

def backward_flatten_and_pool_and_relu(dZ2, W2, P1, pool_mask, Z1):
    # Flatten backward: none -> reshape only
    dF_row = dZ2 * W2                      # (1,4), dZ2/dF = W2
    dP1 = dF_row.reshape(P1.shape)         # reverse the flatten reshape

    # Max-pool backward: route gradient to the recorded argmax cell only
    dA1 = np.zeros_like(Z1)
    for i in range(P1.shape[0]):
        for j in range(P1.shape[1]):
            window_mask = pool_mask[2*i:2*i+2, 2*j:2*j+2]
            dA1[2*i:2*i+2, 2*j:2*j+2] += window_mask * dP1[i, j]

    # ReLU backward: elementwise gate
    dZ1 = dA1 * (Z1 > 0)
    return dP1, dA1, dZ1

def backward_conv(X, dZ1, W1_shape):
    kh, kw = W1_shape
    dW1 = np.zeros(W1_shape)
    for u in range(kh):
        for v in range(kw):
            # cross-correlate X against dZ1, offset by (u,v)
            dW1[u, v] = np.sum(X[u:u+dZ1.shape[0], v:v+dZ1.shape[1]] * dZ1)
    db1 = np.sum(dZ1)
    return dW1, db1
```

Running this on the Part 1 cache reproduces every number in Section 8's table exactly.

## 10. Verifying Against Part 1's Preview and a Framework

Two independent checks were run:

1. **Against Part 1's numerical gradient check.** Part 1 already ran a central-difference
   numerical check on $\partial L/\partial W_1$ and $\partial L/\partial b_1$ (perturbing
   each parameter directly and re-running the *entire* forward pass, with no chain rule
   involved at all) and got agreement to within $10^{-11}$. This part's step-by-step
   derivation reproduces those same values exactly (max difference: $2.2\times10^{-16}$,
   i.e. floating-point rounding only), confirming the layer-by-layer chain rule and the
   all-at-once numerical check agree.
2. **Against the `autograd` framework check from Part 1.** The reverse-mode automatic
   differentiation result computed there for $dW_1$ and $db_1$ also matches this part's
   hand-derived values to the same floating-point precision.

![All gradients summary](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/15-all-gradients-summary.png)
*Figure 7 — With Part 1 and Part 2 combined, all 15 trainable parameters now have a verified
gradient: the mean absolute gradient magnitude per parameter group, for this one example
input. $W_1$ and $W_2$ receive comparable-sized gradients here; in general their relative
scale depends heavily on the specific input, initialization and where in training the network
is — this chart only reflects gradients for the one example computed throughout this series.*

## 11. Setting Up Part 3

All four gradients are now available:

$$\frac{\partial L}{\partial W_1},\quad \frac{\partial L}{\partial b_1},\quad
\frac{\partial L}{\partial W_2},\quad \frac{\partial L}{\partial b_2}$$

Part 3 will use them in the standard gradient-descent update rule with learning rate $\eta$:

$$W_1 \leftarrow W_1 - \eta\frac{\partial L}{\partial W_1}\qquad b_1 \leftarrow b_1 - \eta\frac{\partial L}{\partial b_1}$$
$$W_2 \leftarrow W_2 - \eta\frac{\partial L}{\partial W_2}\qquad b_2 \leftarrow b_2 - \eta\frac{\partial L}{\partial b_2}$$

and will run this update repeatedly on the toy network to show the loss actually decrease
over successive steps — closing the loop from "how do gradients flow backward" (Parts 1–2) to
"how does the network actually learn" (Part 3).

## 12. Key Takeaways

- Flatten has no trainable parameters, so its backward pass is a pure reshape: take
  $\partial L/\partial F$ and reshape it back to $P_1$'s original $(2,2)$ layout.
- Max pooling also has no trainable parameters, but it discards information in the forward
  pass, so its backward pass must recover where each surviving value came from — the
  argmax location recorded during the forward pass. Gradient flows only to that one cell per
  window; every other cell gets exactly zero.
- ReLU's backward pass is a simple elementwise gate: multiply the incoming gradient by 1
  where the original pre-activation was positive, 0 otherwise.
- The convolution layer's weights and bias are **shared** across every output position, so
  their gradients are **sums over all positions** that used them — this is the single
  biggest structural difference from the FC layer in Part 1, where each weight touched only
  one output.
- $\partial L/\partial b_1$ is just the sum of the upstream gradient matrix
  $\partial L/\partial Z_1$.
- $\partial L/\partial W_1$ is the cross-correlation of the original input $X$ with the
  upstream gradient $\partial L/\partial Z_1$ — structurally identical to the forward
  convolution operation, but applied between different tensors, and producing a
  filter-shaped result rather than a feature map.
- Every number derived here matches, to floating-point precision, both the numerical
  gradient check and the automatic-differentiation framework check from Part 1 — three
  independent methods (hand-derived chain rule, finite differences, reverse-mode autodiff)
  now agree on all 15 trainable-parameter gradients.

## 13. Further Reading

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by
  back-propagating errors*. Nature, 323(6088), 533–536.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to
  document recognition*. Proceedings of the IEEE, 86(11), 2278–2324.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 9
  (Convolutional Networks), particularly the section on the convolution gradient as a
  correlation. MIT Press. https://www.deeplearningbook.org/
- Stanford CS231n, backpropagation and CNN gradient notes:
  https://cs231n.github.io/optimization-2/ and https://cs231n.github.io/convolutional-networks/
- Dumoulin, V., & Visin, F. (2016). *A guide to convolution arithmetic for deep learning*.
  arXiv:1603.07285 — covers the relationship between forward convolution and its gradient
  (transposed convolution) in detail.
