# Nesterov Accelerated Gradient (NAG)

Classical momentum fixes gradient descent's slowness by accumulating velocity, but it buys that speed with overshoot: it builds up so much momentum that it sails past the minimum and spends many iterations oscillating around it. Nesterov Accelerated Gradient fixes the overshoot with a single, almost absurdly small change — evaluate the gradient not where you currently are, but where your momentum is *about to take you*. That one-line change acts as a brake that engages automatically as you approach a minimum. This document works through the update rule, the geometry, and the trade-offs, and backs every claim with code that was actually run: on an ill-conditioned quadratic NAG reached the minimum in **39 steps where momentum needed 97**, and on a real neural network it hit the target training loss in **4.2 epochs versus momentum's 6.0**. It also tests two claims that turn out to cut against the simple story — NAG is measurably *less* tolerant of large learning rates than momentum, and it really is worse at escaping local minima.

> **Series note.** This builds directly on [gradient-descent.md](gradient-descent.md) and [momentum.md](momentum.md). If the terms *velocity*, *exponentially weighted average*, or *condition number* are unfamiliar, read those first. Later notes ([adagrad.md](adagrad.md), [rmsprop.md](rmsprop.md), [adam.md](adam.md)) build on this one.

---

## Table of Contents

1. [The problem with classical momentum](#1-the-problem-with-classical-momentum)
2. [The core idea: look before you leap](#2-the-core-idea-look-before-you-leap)
3. [The update rule](#3-the-update-rule)
4. [Geometric intuition: why the look-ahead brakes](#4-geometric-intuition-why-the-look-ahead-brakes)
5. [Experiment: the ill-conditioned quadratic](#5-experiment-the-ill-conditioned-quadratic)
6. [Experiment: measuring the overshoot directly](#6-experiment-measuring-the-overshoot-directly)
7. [Where the simple story breaks: stability](#7-where-the-simple-story-breaks-stability)
8. [Experiment: a real neural network](#8-experiment-a-real-neural-network)
9. [The disadvantage: escaping local minima](#9-the-disadvantage-escaping-local-minima)
10. [What Keras and PyTorch actually implement](#10-what-keras-and-pytorch-actually-implement)
11. [From-scratch implementation](#11-from-scratch-implementation)
12. [Framework usage](#12-framework-usage)
13. [Practical guidance](#13-practical-guidance)
14. [Key takeaways](#14-key-takeaways)
15. [Further reading](#15-further-reading)
16. [Reproducing these results](#16-reproducing-these-results)

---

## 1. The problem with classical momentum

Classical (Polyak, "heavy ball") momentum maintains a velocity vector that is an exponentially weighted accumulation of past gradients:

$$v_t = \beta v_{t-1} - \eta \nabla L(\mathbf{w}_t)$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t + v_t$$

With $\beta = 0.9$, the velocity behaves roughly like an average over the last $\frac{1}{1-\beta} = 10$ gradients, and in a consistently-downhill direction the effective step size approaches $\frac{\eta}{1-\beta}$ — ten times the plain gradient descent step. That is exactly why momentum is fast, and exactly why it overshoots.

The failure mode is mechanical. As the ball rolls into a minimum, the gradient shrinks toward zero. But the velocity does *not* shrink — it is dominated by the large gradients accumulated on the way down. So at the moment the ball reaches the bottom, it is still carrying nearly full speed, and it sails straight through. On the other side the gradient reverses and starts decelerating it, but that deceleration takes several iterations to overcome the stored velocity. The result is a damped oscillation around the minimum.

Measured on an ill-conditioned quadratic (Section 5), momentum travelled a **total path length of 49.3** to cover a straight-line distance of about 9.5 — roughly five times further than necessary — and its first excursion past the minimum reached **4.13 units** on the far side.

## 2. The core idea: look before you leap

NAG's insight: at the start of iteration $t$, you already know a large part of where you are going. The term $\beta v_{t-1}$ is going to be applied regardless of what the gradient says — it is committed. So evaluating the gradient at $\mathbf{w}_t$ is using stale information. You should instead evaluate it at the point momentum is about to carry you to.

Define the **look-ahead point**:

$$\tilde{\mathbf{w}}_t = \mathbf{w}_t + \beta v_{t-1}$$

and take the gradient *there*. The physical analogy: classical momentum is a ball rolling blindly downhill, reacting to the slope under its feet. NAG is a ball that can see a short distance ahead. As it approaches the bottom of a valley, the look-ahead point is already up the opposite wall, where the gradient points *backwards*. So NAG starts braking before it reaches the bottom, while momentum only starts braking after it has passed it.

![One NAG iteration](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_nag_flow.png)

*The five stages of a single NAG iteration. Stages 2 and 3 are the entire contribution of the method: jump to the look-ahead point, and take the gradient there. Stages 4 and 5 are identical to classical momentum — the velocity update and the parameter update are unchanged. This is why NAG costs essentially nothing extra: it is the same number of gradient evaluations per step, just evaluated at a different location.*

## 3. The update rule

Side by side, with the only difference boxed:

**Classical momentum**
$$v_t = \beta v_{t-1} - \eta \nabla L(\mathbf{w}_t)$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t + v_t$$

**NAG**
$$\tilde{\mathbf{w}}_t = \mathbf{w}_t + \beta v_{t-1}$$
$$v_t = \beta v_{t-1} - \eta \nabla L(\tilde{\mathbf{w}}_t)$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t + v_t$$

![Update rules compared](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_update_rules.png)

*The two update rules with the single differing term highlighted. The velocity recursion has the same shape in both — $\beta v_{t-1}$ minus a scaled gradient — and the parameter update is character-for-character identical. The only change is the argument to $\nabla L$: the current position $\mathbf{w}_t$ for momentum, the look-ahead position $\tilde{\mathbf{w}}_t$ for NAG. Everything else in this document follows from that substitution.*

Note the computational cost: one gradient evaluation per iteration, same as momentum. NAG is not a more expensive algorithm — it just points the same computation at a better location.

## 4. Geometric intuition: why the look-ahead brakes

The mechanism is easiest to see by decomposing one step into its two component vectors, starting from an identical state $(\mathbf{w}_t, v_{t-1})$.

![Look-ahead vector decomposition](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_lookahead_vectors.png)

*One step of each method from the same starting state on the quadratic bowl, with real computed vectors. Both begin with the identical committed momentum carry $\beta v_{t-1}$ (amber). They differ in where the gradient correction is measured: momentum measures at $\mathbf{w}_t = (-2.81, 2.35)$ and gets $\nabla L = (-2.81, 47.04)$; NAG measures at the look-ahead point $(-1.53, 2.77)$ and gets $(-1.53, 55.30)$. The look-ahead gradient is both larger in the steep direction and rotated by 1.83°, and the resulting step lands NAG at distance 1.490 from the minimum versus momentum's 1.549.*

Two things are worth noticing in those numbers, because they explain the whole method:

**The correction is stronger where curvature is high.** In the steep $w_2$ direction, the look-ahead gradient component is 55.30 versus 47.04 at the current point — about 18% larger. Momentum is carrying the ball *up* the steep wall, so the look-ahead point is further up that wall, where the restoring force is stronger. NAG therefore applies a bigger brake in exactly the direction that is about to oscillate.

**The correction is weaker where curvature is low.** In the flat $w_1$ direction, the look-ahead gradient is $-1.53$ versus $-2.81$ — *smaller*. Along a direction of consistent gentle slope, moving ahead moves you closer to the minimum, so the gradient there is smaller and NAG brakes *less*. It keeps its speed on the flat axis while damping the steep one.

This asymmetry is the key property. A single mechanism simultaneously suppresses oscillation in high-curvature directions and preserves acceleration in low-curvature ones. On a single step the effect is small — 1.83° of rotation. Compounded over 100 steps it is dramatic, as the next section shows.

Formally, the difference between the two gradients is approximately (first-order Taylor expansion):

$$\nabla L(\mathbf{w}_t + \beta v_{t-1}) \approx \nabla L(\mathbf{w}_t) + \beta \nabla^2 L(\mathbf{w}_t)\, v_{t-1}$$

The extra term is the Hessian applied to the velocity. NAG is therefore *implicitly* using second-order curvature information — it behaves like a correction proportional to $\nabla^2 L \cdot v$ — without ever forming or inverting a Hessian. That is the deeper reason it outperforms momentum on ill-conditioned problems.

## 5. Experiment: the ill-conditioned quadratic

The standard test case. Minimise

$$L(w_1, w_2) = \tfrac{1}{2}\left(w_1^2 + 20 w_2^2\right)$$

a bowl 20× steeper in $w_2$ than $w_1$ (condition number 20). All three methods start at $(-9, 3)$ with identical hyperparameters $\eta = 0.045$, $\beta = 0.9$, and run for 100 steps.

![Trajectories on the quadratic](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_trajectories_quadratic.png)

*Actual computed trajectories, not illustrations. Vanilla GD (left) cannot oscillate but is crippled: it damps the steep direction immediately and then crawls along the flat axis, never arriving. Classical momentum (centre) escapes the flat-direction problem but is visibly chaotic, swinging wildly across the valley and looping around the minimum repeatedly. NAG (right) damps the steep direction just as fast as GD, then travels the flat direction at momentum-like speed, producing a nearly straight path. The final losses differ by five orders of magnitude.*

Measured results after 100 steps:

| Metric | Vanilla GD | Momentum | NAG |
|---|---|---|---|
| Steps to reach $\|w\| < 10^{-2}$ | never (>100) | 97 | **39** |
| Final loss | 4.06e-03 | 3.81e-04 | **4.24e-08** |
| Final distance to minimum | 0.0901 | 0.0270 | **0.000291** |
| Total path length travelled | 11.32 | 49.29 | **19.90** |
| Peak overshoot past minimum ($w_1$) | 0.00 | 4.13 | **2.87** |

NAG reached the convergence threshold in **2.5× fewer steps** than momentum, travelled **60% less total distance**, and ended with a final loss roughly **9,000× lower**. The path-length figure is the clearest single summary of what NAG fixes: momentum covers 49.3 units of ground to make 9.5 units of progress, because most of its motion is sideways oscillation. NAG covers 19.9.

![Loss curves and oscillation](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_loss_and_oscillation.png)

*Left: loss against iteration on a log scale. NAG (blue) descends steadily and keeps descending; momentum (red) descends in a jagged staircase, each plateau corresponding to a swing across the valley where progress stalls. Right: the raw $w_2$ coordinate over time, which is the oscillation itself. Momentum's envelope decays slowly and is still visibly ringing at iteration 100. Note that NAG crosses zero more often but with far smaller amplitude — the oscillation is faster and much more heavily damped, which is what a brake does.*

## 6. Experiment: measuring the overshoot directly

To isolate overshoot from ill-conditioning, here is a 1-D bowl $L(w) = w^2$ starting at $w_0 = -5$, with $\eta = 0.1$, $\beta = 0.9$. There is no flat direction and no conditioning problem — the only thing being measured is how far past the minimum each method flies.

![1-D overshoot](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_overshoot_1d.png)

*Both methods cross the minimum 12 times in 80 iterations, but the amplitudes differ sharply. Momentum's first excursion past zero reaches 3.54 — it travels 71% of the way back up the opposite side of the bowl. NAG's reaches 1.66, less than half as far. More telling is the endgame: NAG settles permanently within 0.05 of the minimum by iteration 28 and finishes at $-6\times10^{-6}$, while momentum never settles within 0.05 at all and is still at $-0.0535$ after 80 iterations.*

| Metric | Momentum | NAG |
|---|---|---|
| Peak overshoot past the minimum | 3.545 | **1.663** |
| Iterations until permanently within 0.05 | never (>80) | **28** |
| Final position after 80 steps | $-5.35\times10^{-2}$ | $-6\times10^{-6}$ |

This is the video's central claim, quantified: NAG cuts the overshoot by **53%** and converts a persistent oscillation into one that actually settles.

## 7. Where the simple story breaks: stability

It is tempting to conclude that NAG is "momentum with brakes, therefore safer". That conclusion is wrong, and it is worth testing because it will bite you when tuning.

The experiment sweeps a grid of $(\eta, \beta)$ pairs on the same quadratic and records whether each run converged or blew up.

![Stability regions](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_stability_regions.png)

*Measured convergence over 300 steps: light = converged, dark = diverged. The amber line is the closed-form linear-stability prediction, the dashed line the measured boundary — they agree almost exactly, confirming the sweep is measuring the real thing. The striking feature is the difference in area: momentum's stable region widens as $\beta$ increases, while NAG's narrows.*

For a quadratic with maximum curvature $L$, the stability limits are:

$$\text{momentum:}\quad \eta < \frac{2(1+\beta)}{L} \qquad\qquad \text{NAG:}\quad \eta < \frac{2(1+\beta)}{L(1+2\beta)}$$

![Critical learning rate](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/08_critical_lr.png)

*The largest stable learning rate for each method as a function of $\beta$. Momentum's rises with $\beta$; NAG's falls. They are identical at $\beta = 0$ (where both reduce to plain gradient descent, limit $2/L$) and diverge steadily after that.*

At $\beta = 0.9$ with $L = 20$, measured:

| | Plain GD | Momentum | NAG |
|---|---|---|---|
| Largest stable $\eta$ (measured) | 0.100 | 0.189 | **0.067** |
| Predicted by formula | 0.100 | 0.190 | 0.068 |

**Momentum tolerates a 2.8× larger learning rate than NAG.** The intuition is that the look-ahead is itself an extrapolation, and extrapolating from an already-too-large step amplifies the error rather than correcting it. The brake metaphor holds near a minimum but fails at the edge of stability.

The practical consequence: **when you switch from `nesterov=False` to `nesterov=True`, do not assume your learning rate transfers.** If you were already near the edge, NAG will diverge where momentum did not. This is not a hypothetical — it happened during the neural network experiment in the next section.

## 8. Experiment: a real neural network

Everything so far has been on quadratics. Here is a genuine (if small) deep learning task: a 64→64→32→10 ReLU MLP with softmax cross-entropy, trained from scratch in NumPy on scikit-learn's handwritten digits dataset (1,797 samples, 8×8 images), 25% held out for test, mini-batch size 64, 5 random seeds per configuration.

First, a learning-rate sweep, because comparing optimisers at a single shared learning rate is a badly flawed methodology — it just measures which method happens to like that particular value.

![Learning rate sweep](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/11_lr_sweep.png)

*Training loss after 40 epochs against learning rate, median of 3 seeds. Both momentum methods dominate plain SGD across the entire usable range — at $\eta=0.05$ they reach a loss ~35× lower. But the right-hand edge shows the stability finding from Section 7 reappearing on a real network: at $\eta = 0.25$ momentum still trains (loss 0.071) while NAG has already collapsed (loss 1.74). NAG's optimum is slightly lower and its cliff noticeably earlier.*

This sweep also caught a bug in the making. My first attempt ran all three methods at $\eta = 0.25$, and both momentum and NAG produced garbage — final training losses near 1.0 with test accuracy in the 40–60% range. That is not a result about NAG; it is a result about an effective step size of $\eta/(1-\beta) = 2.5$. At $\eta = 0.12$, one NAG seed out of five destabilised late in training (final loss 0.303 versus ~0.0003 for the other four) while momentum stayed stable on all five — the same asymmetry again. The head-to-head below therefore uses $\eta = 0.08$ for both momentum methods, comfortably inside the stable region for both, and $\eta = 0.25$ for plain SGD, its own best stable value.

![MLP training curves](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/09_mlp_digits.png)

*Mean over 5 seeds, shaded band showing min–max across seeds. NAG (blue) is clearly ahead of momentum (red) for the first ~15 epochs — it reaches any given loss level sooner — after which the two curves merge and are indistinguishable. Plain SGD (grey) is both slower and dramatically noisier, its min–max band spanning orders of magnitude. On test accuracy, NAG reaches the ~97% plateau within about 6 epochs, momentum takes ~10, SGD takes ~22.*

![Epochs to threshold](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/10_epochs_to_threshold.png)

*Mean epochs required to first reach a training loss of 0.10. NAG needed 4.2, momentum 6.0, plain SGD 10.6 — NAG is 30% faster than momentum and 2.5× faster than SGD to this milestone. Per-seed values were NAG [5, 3, 4, 4, 5] and momentum [6, 5, 7, 7, 5], so the gap is consistent across every seed rather than driven by an outlier.*

| Metric (5 seeds) | Plain SGD ($\eta$=0.25) | Momentum ($\eta$=0.08) | NAG ($\eta$=0.08) |
|---|---|---|---|
| Epochs to train loss 0.10 | 10.6 | 6.0 | **4.2** |
| Final train loss | 5.70e-03 ± 9.6e-04 | **5.93e-04** ± 5.3e-05 | 6.40e-04 ± 6.5e-05 |
| Final test accuracy | 97.20% ± 0.27 | **97.42%** ± 0.36 | 97.33% ± 0.31 |
| Best test accuracy | 97.82% | 97.69% | **97.91%** |

**Read this table honestly.** NAG's advantage is real but narrow. It is a clear and consistent win on *convergence speed* — 30% fewer epochs to the target loss, on every seed. It is a **tie on final quality**: the 0.09-point test accuracy gap between momentum and NAG is well inside the ±0.3 seed-to-seed noise, and the final training losses are within 8% of each other. Anyone claiming NAG produces better models on a task like this is over-reading the data. The honest summary is: same destination, fewer epochs to get there.

The gap would be expected to widen on harder, more ill-conditioned problems — the quadratic in Section 5 had a condition number of 20 and showed a 2.5× speedup — and to shrink on easy, well-conditioned ones like this.

## 9. The disadvantage: escaping local minima

The video's stated drawback is that NAG's damping makes it worse at escaping local minima. This is testable, so here is the test.

The landscape is an asymmetric double well, $L(w) = 0.05w^4 - 0.6w^2 + 0.55w$, which has a shallow local minimum at $w = 2.176$ (depth $-0.523$), a barrier at $w = 0.476$ (height $0.128$), and the global minimum at $w = -2.653$ (depth $-3.205$). The barrier stands $0.652$ above the local minimum. Each run starts at $w = 4.176$, up the outer wall, so the ball has a run-up: it rolls down through the shallow basin and either carries enough momentum over the barrier or does not. $\eta = 0.02$, 800 steps, sweeping $\beta$ from 0 to 0.99.

![Local minima escape](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/12_local_minima_escape.png)

*Left: the test landscape with the start point and the path the ball must take. Right: escape outcome against $\beta$, zoomed on the interesting region. Momentum escapes the shallow basin once $\beta \geq 0.87$; NAG requires $\beta \geq 0.90$. In the shaded band ($\beta$ = 0.87, 0.88, 0.89) momentum reaches the global minimum and NAG does not — it gets braked at the barrier and falls back.*

![Escape trajectories](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/13_escape_trajectories.png)

*The two trajectories at $\beta = 0.88$, the middle of the differing band. Momentum (red) crests the barrier and settles at the global minimum $w = -2.653$. NAG (blue) approaches the barrier, gets slowed by the look-ahead gradient — which, at the barrier's near side, points backwards — and falls back into the shallow basin, ending at $w = 2.176$. Same landscape, same start, same $\eta$, same $\beta$; the only difference is where the gradient was measured.*

**The claim holds.** The mechanism is exactly the one that makes NAG good at minima: as the ball climbs toward the barrier crest, the look-ahead point is further up the slope where the gradient is more strongly opposing, so NAG brakes. Braking is what you want near a minimum you wish to settle into, and precisely what you do not want when the "minimum" is a shallow trap you need to escape. You cannot have one without the other — they are the same mechanism.

Two important caveats before generalising this:

- **This is one hand-built 1-D landscape.** It demonstrates the mechanism convincingly; it does not establish a universal law. The size of the gap (0.87 vs 0.90) depends on the run-up distance, $\eta$, and the barrier geometry — across the settings I swept, the gap ranged from 0.01 to 0.04 in $\beta$, but momentum escaped at an equal or lower $\beta$ in every single one.
- **Local minima are largely the wrong worry for real deep networks.** Dauphin et al. (2014) argue that in high-dimensional loss surfaces the dominant obstacle is saddle points, not poor local minima — a point requires *all* of its many curvature directions to turn upward to be a local minimum, which becomes exponentially unlikely as dimension grows. So while this disadvantage is real and measurable, it is not usually the reason NAG would fail you on a real network. The stability finding in Section 7 is the more practically dangerous one.

## 10. What Keras and PyTorch actually implement

Here is something the textbook presentation obscures, and a common source of confusion when people compare their from-scratch implementation against a framework and find the weights do not match.

Neither Keras nor PyTorch implements the update rule from Section 3. Both implement a reformulation (due to Sutskever et al., 2013) in which the variable being tracked is the **look-ahead point itself** rather than $\mathbf{w}_t$. This is done because the textbook form requires evaluating the gradient at a point that is not the current parameter vector — awkward inside a framework where the optimiser only sees parameters and their gradients.

Substituting $\theta_t = \mathbf{w}_t + \beta v_{t-1}$ into the textbook rule gives:

$$v_t = \beta v_{t-1} - \eta \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t + (1+\beta)v_t - \beta v_{t-1}$$

Now the gradient is evaluated at the tracked variable, which is exactly what a framework optimiser can do. **Keras** implements this as `v = β·v - lr·g; x = x + β·v - lr·g`, and **PyTorch** as `buf = μ·buf + g; d = g + μ·buf; x = x - lr·d`. These look different from each other and from the equation above; all three are algebraically identical.

![Framework equivalence](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/15_framework_equivalence.png)

*Numerical verification on a random 5-dimensional convex quadratic over 200 iterations. Left: the Keras form and the PyTorch form each track the textbook method's look-ahead sequence $\tilde{\mathbf{w}}_t$ to a maximum absolute difference of $4.4\times10^{-16}$ — machine precision, i.e. they are the same algorithm. The red line shows the same Keras form compared against the textbook $\mathbf{w}_t$ sequence instead, differing by 0.103 — confirming the frameworks genuinely track a different (shifted) variable, not merely a rounding-level variant. Right: both descend identically.*

Verified results:

| Comparison | Max absolute difference |
|---|---|
| Keras form vs look-ahead sequence $\tilde{\mathbf{w}}_t$ | 4.44e-16 |
| PyTorch form vs look-ahead sequence $\tilde{\mathbf{w}}_t$ | 4.44e-16 |
| Keras form vs PyTorch form | 3.33e-16 |
| Keras form vs textbook sequence $\mathbf{w}_t$ | **0.103** |

**Practical implication:** if you implement textbook NAG and compare weights against Keras step by step, they will not match — and nothing is wrong. Your $\mathbf{w}_t$ and their $\theta_t$ differ by exactly $\beta v_{t-1}$, one look-ahead. Both converge to the same minimum. Compare losses, not weights.

## 11. From-scratch implementation

Complete NumPy implementation, verified to run. The design point worth noting is that `lookahead()` and `step()` are separate calls, because the caller must compute gradients *between* them — this is structurally different from a normal optimiser API where you just hand over gradients. Note also how `Momentum` is derived from `NAG` by overriding a single method to return the parameters unchanged: that is the entire difference between the two algorithms, expressed as code.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class NAG:
    """Nesterov Accelerated Gradient.

    Usage per step:
        look = opt.lookahead(params)      # where to evaluate the gradient
        grads = compute_grads(look)       # <- gradient taken AT the look-ahead
        params = opt.step(params, grads)
    """

    def __init__(self, lr=0.08, beta=0.9):
        self.lr, self.beta, self.v = lr, beta, None

    def _init(self, params):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]

    def lookahead(self, params):
        self._init(params)
        return [p + self.beta * v for p, v in zip(params, self.v)]

    def step(self, params, grads):
        self._init(params)
        self.v = [self.beta * v - self.lr * g for v, g in zip(self.v, grads)]
        return [p + v for p, v in zip(params, self.v)]


class Momentum(NAG):
    """Classical momentum: identical, except the gradient is taken where we stand."""

    def lookahead(self, params):
        self._init(params)
        return params


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
        order = rng.permutation(len(Xtr))          # one shuffle per epoch
        for s in range(0, len(Xtr), bs):
            idx = order[s:s + bs]
            look = opt.lookahead(P)                          # 1. jump ahead
            probs, caches = forward(look, Xtr[idx])          # 2. forward THERE
            grads = backward(look, caches, probs, Ytr[idx])  # 3. gradient THERE
            P = opt.step(P, grads)                           # 4. velocity + move
    return evaluate(P, Xte, yte)


if __name__ == "__main__":
    for name, opt in [("Momentum", Momentum(lr=0.08, beta=0.9)),
                      ("NAG     ", NAG(lr=0.08, beta=0.9))]:
        loss, acc = train(opt)
        print(f"{name}  test loss {loss:.4f}   test accuracy {acc*100:.2f}%")
```

Actual output when run:

```
Momentum  test loss 0.1307   test accuracy 97.11%
NAG       test loss 0.1228   test accuracy 97.33%
```

The backward pass was verified against numerical differentiation (central differences, $\epsilon = 10^{-6}$, 48 randomly sampled parameters across four weight and bias tensors): **maximum relative error $1.93\times10^{-7}$**, which is the expected floor for float64 central differences. The gradients are correct.

## 12. Framework usage

NAG is not a separate optimiser class in either framework — it is a flag on SGD.

**Keras:**

```python
from tensorflow.keras.optimizers import SGD

optimizer = SGD(learning_rate=0.08, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer, loss="categorical_crossentropy",
              metrics=["accuracy"])
```

**PyTorch:**

```python
import torch

optimizer = torch.optim.SGD(model.parameters(), lr=0.08,
                            momentum=0.9, nesterov=True)
```

Gotchas worth knowing:

- `nesterov=True` **requires** `momentum > 0`. Setting `nesterov=True, momentum=0` raises an error in PyTorch and is meaningless in both (with no velocity there is nothing to look ahead along).
- PyTorch additionally requires `dampening=0` when `nesterov=True`.
- These two snippets were **not executed** — neither framework is installed in the environment used for this document. Everything else in these notes was run. The update equations they implement, however, *were* verified numerically in Section 10 by reimplementing both framework update rules in NumPy.

## 13. Practical guidance

- **Default to `nesterov=True`** when you are using SGD with momentum. It costs nothing extra per step and is at worst neutral.
- **Re-tune your learning rate when you enable it.** This is the one thing most likely to bite you. Per Section 7, NAG's stable learning-rate ceiling is roughly $2.8\times$ lower than momentum's at $\beta = 0.9$. If you flip the flag and training suddenly diverges, lower $\eta$ before concluding NAG is broken.
- **$\beta = 0.9$ remains a sensible default**; 0.95 or 0.99 pair well with longer schedules and smaller $\eta$.
- **Expect faster convergence, not a better final model.** On the network here, NAG reached the target loss 30% sooner and finished in a statistical tie on accuracy. Budget your expectations accordingly.
- **The benefit scales with ill-conditioning.** On the condition-number-20 quadratic NAG was 2.5× faster; on the well-conditioned small MLP it was 1.4× faster. If your loss surface has wildly varying curvature, NAG helps more.
- **For adaptive optimisers, the analogue is NAdam** (Dozat, 2016), which grafts the same look-ahead onto Adam. See [adam.md](adam.md).

## 14. Key takeaways

![Summary of all measurements](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/14_summary_table.png)

*Every measurement in this document in one place, collated automatically from the JSON output of the experiment scripts rather than typed by hand. Reading down the NAG column: it wins decisively on convergence speed and overshoot, ties on final model quality, and loses on both maximum stable learning rate and local-minimum escape. That mixed picture is the accurate summary of the method.*

1. **The entire method is one substitution.** Evaluate $\nabla L$ at $\mathbf{w}_t + \beta v_{t-1}$ instead of at $\mathbf{w}_t$. Velocity update and parameter update are unchanged, and the cost per iteration is identical.
2. **It works because the look-ahead senses curvature.** Measured on a real step, the look-ahead gradient was 18% *larger* in the steep direction and 46% *smaller* in the flat one. It brakes hard where oscillation threatens and stays fast where progress is safe — effectively a cheap second-order correction, since the first-order expansion of the look-ahead gradient contains a $\beta\nabla^2 L\, v$ term.
3. **The speedup is real and measured.** 39 vs 97 steps on the ill-conditioned quadratic; 4.2 vs 6.0 epochs on a real MLP; total path length cut by 60%; peak overshoot cut by 53% in the 1-D test.
4. **It reduces overshoot but does not raise the stability ceiling.** At $\beta = 0.9$, momentum tolerated $\eta = 0.189$ and NAG only $0.067$ — momentum takes 2.8× larger steps before diverging. Measured on both a quadratic and a real network. This contradicts the naive "brakes make it safer" reading and is the single most practically important caveat.
5. **The local-minimum disadvantage is genuine.** Momentum escaped a test barrier from $\beta \geq 0.87$; NAG needed $\beta \geq 0.90$. The braking that helps at real minima also brakes at barriers you want to clear. It is the same mechanism — you cannot keep one and discard the other.
6. **Frameworks implement a shifted reformulation.** Keras and PyTorch both track the look-ahead point, not $\mathbf{w}_t$. Verified identical to textbook NAG to $4.4\times10^{-16}$, while differing from the textbook $\mathbf{w}_t$ sequence by 0.103. Do not expect weight-for-weight agreement with a from-scratch implementation.
7. **Gains are largest on ill-conditioned problems** and shrink toward nil on easy, well-conditioned ones.

## 15. Further reading

- **Nesterov, Y. (1983).** *A method of solving a convex programming problem with convergence rate $O(1/k^2)$.* Soviet Mathematics Doklady, 27(2), 372–376. — The original result. Establishes the $O(1/k^2)$ rate for convex functions, against gradient descent's $O(1/k)$.
- **Polyak, B. T. (1964).** *Some methods of speeding up the convergence of iteration methods.* USSR Computational Mathematics and Mathematical Physics, 4(5), 1–17. — The heavy-ball method, i.e. the classical momentum that NAG improves on.
- **Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013).** *On the importance of initialization and momentum in deep learning.* ICML 2013, PMLR 28(3), 1139–1147. — Introduced the reformulation that Keras and PyTorch implement, and made the case for NAG in deep networks specifically.
- **Nesterov, Y. (2004).** *Introductory Lectures on Convex Optimization: A Basic Course.* Springer. — Textbook treatment with the full convergence proofs.
- **Dauphin, Y., Pascanu, R., Gulcehre, C., Cho, K., Ganguli, S., & Bengio, Y. (2014).** *Identifying and attacking the saddle point problem in high-dimensional non-convex optimization.* NeurIPS 2014. — Why the local-minima disadvantage in Section 9 matters less than it sounds for real networks.
- **Su, W., Boyd, S., & Candès, E. (2014).** *A differential equation for modeling Nesterov's accelerated gradient method.* NeurIPS 2014. — Derives the continuous-time ODE limit of NAG; the cleanest available explanation of *why* acceleration happens.
- **Dozat, T. (2016).** *Incorporating Nesterov momentum into Adam.* ICLR 2016 Workshop. — NAdam, the adaptive-optimiser analogue.
- **Goh, G. (2017).** *Why Momentum Really Works.* Distill. — Interactive visual treatment of momentum and conditioning. Excellent companion to Sections 5–7.
