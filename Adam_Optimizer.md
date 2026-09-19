# Adam: Adaptive Moment Estimation

Adam is the default optimizer in most deep-learning code, and the reason is narrow and
specific: it keeps two running averages per parameter — one of the gradient, one of the
squared gradient — and divides the first by the square root of the second. That single
division makes the step size scale-free, so a parameter whose gradients are tiny and a
parameter whose gradients are huge both move about `η` per step. These notes derive the
update rule, and then test each of the claims usually made for it. Every figure was
produced by running code; every number in the prose is output from an experiment included
in full at the end. Three of the tests came out against the usual story, and those are the
interesting ones.

This builds directly on [momentum-optimization.md](momentum-optimization.md), which covers
the first-moment half of Adam in detail; the exponentially weighted moving average, the
`1/(1-β)` effective window, and the coupling between `β` and the learning rate are all
developed there and used here without re-deriving them. See also
[gradient-descent.md](gradient-descent.md) and [sgd.md](sgd.md).

---

## Table of Contents

1. [Where Adam comes from](#1-where-adam-comes-from)
2. [The second moment: AdaGrad and RMSProp](#2-the-second-moment-adagrad-and-rmsprop)
3. [The Adam update rule](#3-the-adam-update-rule)
4. [Bias correction, measured](#4-bias-correction-measured)
5. [What adaptivity actually buys: the rotation test](#5-what-adaptivity-actually-buys-the-rotation-test)
6. [Choosing beta1 and beta2](#6-choosing-beta1-and-beta2)
7. [Sparse features](#7-sparse-features)
8. [A non-convex benchmark](#8-a-non-convex-benchmark)
9. [The bounded step, and why Adam is slow to high precision](#9-the-bounded-step-and-why-adam-is-slow-to-high-precision)
10. [When Adam fails to converge](#10-when-adam-fails-to-converge)
11. [A real network, end to end](#11-a-real-network-end-to-end)
12. [Implementation from scratch](#12-implementation-from-scratch)
13. [Practical guidance](#13-practical-guidance)
14. [Key takeaways](#14-key-takeaways)
15. [Further reading](#15-further-reading)
16. [Reproducing these results](#16-reproducing-these-results)

---

## 1. Where Adam comes from

Two independent lines of work lead to Adam. One asks *which direction* to step and answers
"a smoothed average of recent gradients" — that is momentum, and its successor NAG. The
other asks *how big* a step each individual parameter should take and answers "divide by
the size of that parameter's recent gradients" — that is AdaGrad, and its successor
RMSProp. Adam takes one idea from each.

![Optimizer family tree](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_optimizer_family_tree.png)

*The lineage. The upper branch tracks the first moment of the gradient (its mean) and
controls direction; the lower branch tracks the second raw moment (its mean square) and
controls per-parameter scale. Adam runs both at once and adds a correction for the fact
that both averages start at zero. The dashed arrow is a reminder that Adam uses plain
momentum, not the look-ahead gradient — combining NAG with Adam gives NAdam, a different
algorithm.*

The distinction that matters throughout: the first moment is a **vector** operation that
does the same thing to every coordinate, while the second moment is a **per-coordinate**
operation that treats each parameter separately. That asymmetry is the source of both
Adam's strengths and its one structural weakness, measured in
[section 5](#5-what-adaptivity-actually-buys-the-rotation-test).

---

## 2. The second moment: AdaGrad and RMSProp

**AdaGrad** (Duchi et al., 2011) accumulates every squared gradient it has ever seen:

$$s_t = s_{t-1} + g_t^2, \qquad w_t = w_{t-1} - \frac{\eta}{\sqrt{s_t}+\epsilon}\, g_t$$

All operations are elementwise. A parameter that has received large gradients gets a small
step; one that has received little signal keeps a large step. The problem is that $s_t$ is
monotonically non-decreasing, so the effective step size $\eta/\sqrt{s_t}$ can only shrink.
If the gradients do not die away, $s_t$ grows roughly linearly in $t$ and the step decays
like $t^{-1/2}$ — eventually to nothing, whether or not the optimizer has arrived.

**RMSProp** (Tieleman & Hinton, 2012) replaces the sum with an exponentially weighted
average, the same primitive momentum uses:

$$s_t = \rho\, s_{t-1} + (1-\rho)\, g_t^2, \qquad w_t = w_{t-1} - \frac{\eta}{\sqrt{s_t}+\epsilon}\, g_t$$

Now $s_t$ tracks the *recent* gradient scale and can go down as well as up.

**Experiment (measured).** On the 20-dimensional quadratic used throughout these notes,
with persistent Gaussian gradient noise ($\sigma = 0.5$) and both methods at the same
$\eta = 0.01$, tracking the flattest coordinate over 3000 steps:

| | effective step at $t=1$ | at $t=3000$ | shrink factor | mean loss over last 200 steps |
|---|---|---|---|---|
| AdaGrad | 0.0259 | 0.000352 | **73.6×** | **512.7** |
| RMSProp | 0.8195 | 0.0201 | 40.7× | **0.018** |

Fitting a power law to AdaGrad's effective step over steps 100–3000 gives an exponent of
**−0.485**, against the predicted $t^{-1/2}$. AdaGrad's step decayed faster than the
optimizer could make progress, and after 3000 steps its loss was still four orders of
magnitude above RMSProp's.

![AdaGrad vs RMSProp](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_adagrad_vs_rmsprop.png)

*(a) The accumulators on log-log axes. AdaGrad's running sum climbs without bound;
RMSProp's running average flattens once it has seen a representative sample of gradient
magnitudes. (b) The effective step, which is what the parameter actually feels — AdaGrad's
straight line on log-log axes is the $t^{-0.485}$ decay. (c) The consequence. Note the
setup detail that makes this visible: with **noiseless** gradients on a convex quadratic
the gradient vanishes as $w \to 0$, so AdaGrad's sum stops growing and the decay never
bites. An earlier version of this experiment ran without noise and showed AdaGrad
converging perfectly well, which is true and completely unrepresentative of mini-batch
training, where gradient noise persists forever.*

---

## 3. The Adam update rule

Adam (Kingma & Ba, 2015) runs both averages and divides:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad \text{(first moment — direction)}$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \qquad \text{(second raw moment — scale)}$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{\,t}} \qquad \text{(bias correction)}$$

$$w_t = w_{t-1} - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

with defaults $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.
All operations are elementwise, so each parameter gets its own scale.

![The Adam update step](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_adam_update_diagram.png)

*The dataflow. The gradient feeds two independent running averages, each is corrected for
its zero initialization, and the update is their ratio. Setting $\beta_2 = 0$ and removing
the division recovers momentum in its EWMA form; setting $\beta_1 = 0$ recovers RMSProp
with bias correction. Everything Adam does is in those two branches.*

Note that $\hat{m}_t / \sqrt{\hat{v}_t}$ is a **signal-to-noise ratio**. If a coordinate's
gradients are consistently signed, $|\hat{m}| \approx \sqrt{\hat{v}}$ and the ratio is near
$\pm 1$. If they alternate in sign, $\hat{m} \to 0$ while $\hat{v}$ stays large and the
ratio collapses. Adam therefore steps confidently where the gradient is consistent and
cautiously where it is noisy, which is the property that makes it robust to a badly chosen
$\eta$ — and also, as [section 9](#9-the-bounded-step-and-why-adam-is-slow-to-high-precision)
shows, the property that makes it slow to reach high precision.

One consequence worth stating explicitly: because the ratio is dimensionless, multiplying
the loss by 1000 does not change Adam's trajectory at all, whereas it multiplies every SGD
step by 1000. Adam is invariant to rescaling of the loss; SGD is not.

---

## 4. Bias correction, measured

Both averages start at $m_0 = v_0 = 0$, so both are biased toward zero early on. Unrolling
the recursion under a stationary gradient shows the size of the bias exactly:

$$\mathbb{E}[v_t] = (1-\beta_2)\sum_{k=1}^{t}\beta_2^{\,t-k}\,\mathbb{E}[g_k^2] \approx \mathbb{E}[g^2]\,(1-\beta_2^{\,t})$$

which is why dividing by $1-\beta_2^{\,t}$ removes it, and likewise for $m$.

The naive expectation is that skipping bias correction makes early steps too *small*. It
does not, and the reason is that the two biases do not cancel — they compound in opposite
directions, because $\hat m$ appears linearly and $\hat v$ under a square root.

**Experiment (measured).** Feed a constant gradient $g = 1$ with $\beta_1 = 0.9$,
$\beta_2 = 0.999$:

| | uncorrected | corrected | true value |
|---|---|---|---|
| $m_1$ | 0.1 | 1.0 | 1.0 |
| $\sqrt{v_1}$ | 0.0316 | 1.0 | 1.0 |
| step size at $t=1$ | $3.162\,\eta$ | $\eta$ | — |

The first moment is 10× too small, but the square root of the second moment is
$\sqrt{0.001} = 0.0316$, i.e. **31.6×** too small, so their ratio is 3.16× too *large*. The
error peaks at **6.569×** at step **12**, and takes until step **3925** to come within 1% of
the corrected value, because $\beta_2 = 0.999$ has an effective window of 1000 steps.

![Bias correction](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_bias_correction.png)

*(a) The first moment warms up over roughly $1/(1-\beta_1) = 10$ steps. (b) The second
moment, on a log time axis, takes about a thousand. (c) The ratio of the uncorrected step
to the corrected one, which is what the parameters actually experience: not a gentle
warm-up but a 6.6× overshoot that persists for hundreds of steps. Without correction, Adam
takes its largest steps precisely when it knows least about the loss surface — which is
also the argument for learning-rate warmup even when correction is on.*

---

## 5. What adaptivity actually buys: the rotation test

This is the experiment worth doing before believing anything else about adaptive methods.
Take a 20-dimensional quadratic with eigenvalues log-spaced from 1 to 100 ($\kappa = 100$),
and run it two ways:

- **aligned**: the Hessian is diagonal, so each parameter corresponds to one eigendirection;
- **rotated**: the same eigenvalues, expressed in a random orthogonal basis.

These are the *same optimization problem*. Rotating the coordinate system cannot change how
hard a problem is in any meaningful sense — and SGD, momentum, and NAG are provably
invariant to it, because their update is a scalar multiple of a vector. Per-coordinate
methods have no such guarantee, because "per coordinate" is a statement about the basis.

Learning rates were tuned by grid search for every method and both bases, scored by
worst-case steps over 8 random starts to reach $\|w\| < 10^{-3}\|w_0\|$ and stay there:

| optimizer | aligned | rotated | rotation penalty |
|---|---|---|---|
| SGD | 298 | 312 | **1.0×** |
| Momentum | 127 | 126 | **1.0×** |
| NAG | 100 | 102 | **1.0×** |
| AdaGrad | 8 | 376 | **47×** |
| RMSProp | **4** | 5695 | **1424×** |
| Adam | 1230 | 3238 | 2.6× |

![Quadratic race](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_quadratic_race.png)

*(a) On the axis-aligned problem the adaptive methods are spectacular — RMSProp reaches
tolerance in 4 steps — because a diagonal Hessian means a diagonal preconditioner is the
exact inverse Hessian up to scale. They are effectively doing Newton's method for free.
(b) Rotate the basis and that gift disappears: RMSProp needs 5695 steps for the identical
problem, while the three non-adaptive methods do not move at all. The curves are single
runs from one start at the tuned $\eta$; the legend numbers are worst-case over all 8.*

The lesson is not that adaptive methods are bad. It is that **their advantage comes from the
loss surface being approximately axis-aligned in the parameterization you happen to be
using**, and the amount of that advantage is an empirical property of your model, not a
guarantee of the algorithm. Deep networks do tend to have loosely axis-aligned curvature —
different layers, and different units within a layer, genuinely do operate at different
scales — which is the real reason adaptive methods help there. This experiment says that
reason, not "Adam is faster", is what you should expect to hold.

A second measurement from the same runs: over the first 50 steps on the aligned problem,
the ratio between the largest and smallest per-coordinate step was **5.7** for Adam and
about $1.5\times10^{13}$ for SGD. SGD's step in each direction is proportional to that
direction's gradient, so it spans the full range of curvatures; Adam compresses that range
to a single order of magnitude. That compression *is* the algorithm.

---

## 6. Choosing beta1 and beta2

**Experiment (measured).** On the aligned 20-D quadratic, sweeping both decay rates and
re-tuning $\eta$ by grid search in every cell:

![Beta grid](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_beta_grid.png)

*Worst-case steps to tolerance over 8 starts. The best cell is $\beta_1 = 0.9$,
$\beta_2 = 0.9$ at 208 steps; the worst is $\beta_1 = 0$, $\beta_2 = 0.9999$ at 5545 — a
spread of 26.7× from the decay rates alone, with the learning rate already optimized at
every point. The library defaults (orange box) give 1294 steps here, six times the best
cell on this particular problem. One cell, $\beta_1 = 0.99$ with $\beta_2 = 0.9$, never
reached tolerance at any learning rate in the grid.*

Two things to read off this. First, $\beta_1$ behaves as it does for plain momentum — 0.9
is a good value, 0 is worse, 0.99 is unstable — and the reasoning in
[momentum-optimization.md](momentum-optimization.md) carries over unchanged. Second,
$\beta_2$ is doing something different from $\beta_1$ and the default 0.999 is **not**
optimal here. A large $\beta_2$ means the denominator reflects gradient magnitudes from a
thousand steps ago; on a deterministic problem whose gradients shrink steadily, that stale
denominator is systematically too large and the steps are too small.

The defaults are tuned for the opposite regime: mini-batch gradients where $g_t^2$ is a
noisy estimate of the true squared gradient and needs heavy averaging to be usable. On a
noiseless benchmark that averaging is pure lag. Treat the defaults as a good prior for
noisy problems, not as a fact about the algorithm.

---

## 7. Sparse features

The original case for per-parameter scaling was sparse data: a feature present in 0.5% of
examples contributes to the gradient 200× less often than one present in every example, so
under SGD its weight is learned 200× more slowly.

**Experiment (measured).** Logistic regression, 60 features whose presence probabilities are
log-spaced from 1.0 down to 0.005 (26 features with $p < 0.05$, 8 with $p > 0.5$), 6000
samples, mini-batches of 64, learning rate tuned per optimizer:

| optimizer | steps to reach full-batch loss 0.29 | rare-feature error at step 500 | at step 4000 |
|---|---|---|---|
| SGD | 820 | 0.7106 | 0.2843 |
| Momentum | 880 | 0.6558 | 0.2802 |
| AdaGrad | **160** | **0.3664** | 0.2923 |
| RMSProp | **160** | 0.3699 | 0.2973 |
| Adam | 340 | 0.391 | 0.3067 |

![Sparse features](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_sparse_features.png)

*(a) The adaptive methods reach the loss threshold about five times sooner. (b) Where the
difference lives: relative error on the rare features, which is roughly halved at step 500.
The claim is real, and it is specifically a claim about **speed**, not about the final
answer — by step 4000 every method has converged to essentially the same solution, and
SGD's final rare-feature error (0.2843) is in fact marginally the best of the six. If the
budget is unlimited, adaptivity buys nothing here; if the budget is 500 steps, it buys a
factor of two.*

---

## 8. A non-convex benchmark

The Beale function is a standard non-convex test surface with a long curved valley:

$$L(w_1,w_2) = (1.5 - w_1 + w_1 w_2)^2 + (2.25 - w_1 + w_1 w_2^2)^2 + (2.625 - w_1 + w_1 w_2^3)^2$$

with its minimum at $(3, 0.5)$. Starting from $(-1, 1.6)$, learning rate tuned per method,
counting steps to reach and stay within 0.05 of the minimum:

| optimizer | best $\eta$ | steps |
|---|---|---|
| SGD | 0.0411 | 171 |
| Momentum | 0.0096 | 152 |
| AdaGrad | — | **never reached it** at any $\eta$ in the grid within 4000 steps |
| RMSProp | 0.0549 | 460 |
| Adam | 0.8241 | **108** |

![Beale function](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/09_beale_nonconvex.png)

*(a) Trajectories on the log-loss contours. The paths are genuinely erratic — this surface
has regions where the gradient spans many orders of magnitude, which is exactly where a
scale-free step helps. (b) Loss against step. Adam wins, and note the learning rate it won
with: 0.82, some twenty times larger than SGD's. Because Adam's step is bounded by roughly
$\eta$ regardless of gradient size ([section 9](#9-the-bounded-step-and-why-adam-is-slow-to-high-precision)),
a large $\eta$ that would make SGD explode here is safe for Adam. AdaGrad's failure is the
same decay problem as [section 2](#2-the-second-moment-adagrad-and-rmsprop): the early
steep region inflates its accumulator permanently, and the step size never recovers.*

---

## 9. The bounded step, and why Adam is slow to high precision

Kingma & Ba prove that Adam's per-step displacement is bounded:

$$|\Delta w_t| \le \eta \cdot \frac{1-\beta_1}{\sqrt{1-\beta_2}} \quad \text{when } (1-\beta_1) > \sqrt{1-\beta_2}, \qquad |\Delta w_t| \le \eta \ \text{ otherwise}$$

With the defaults, $(1-\beta_1)/\sqrt{1-\beta_2} = 0.1/0.0316 = 3.162$, so the ceiling is
$3.162\,\eta$, and in the steady state where $|\hat m| \approx \sqrt{\hat v}$ it sits at
about $\eta$.

**Experiment (measured).** Recording the largest per-parameter change on each of 480
mini-batch updates while training the network from
[section 11](#11-a-real-network-end-to-end):

| | median $\max_i|\Delta w_i| / \eta$ | maximum |
|---|---|---|
| Adam | 1.001 | **1.871** (bound: 3.162) |
| SGD | 0.155 | 2.6 |

![Step size bound](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/11_step_size_bound.png)

*(a) Adam's largest displacement per step, in units of $\eta$, stays in a narrow band around
1 and never approaches the theoretical ceiling. (b) SGD's equivalent quantity is just the
largest gradient component, and it spans a 16.5× range between its median and its maximum
in the same run — note the log-scaled counts. This is why an $\eta$ that works for Adam
tends to keep working as training progresses, and why a single $\eta$ for SGD often does
not.*

The bound has a cost, visible in the rotation table in
[section 5](#5-what-adaptivity-actually-buys-the-rotation-test): on the aligned quadratic
Adam needed 1230 steps where momentum needed 127. It is not that Adam is badly behaved
there — it is arithmetic. If each step moves at most about $\eta$ and the trajectory must
cover a distance of 10, then at least $10/\eta$ steps are required; and $\eta$ must also be
small enough that the optimizer can settle inside a tolerance of 0.01. Those two demands
together force roughly 1000 steps. Scoring the same runs at a looser tolerance of 1.0
instead of 0.01 flips the picture: Adam takes **34** steps against momentum's 40 and SGD's
69, on the same problem with $\eta$ re-tuned.

So Adam gets to a good-enough solution quickly and to a very precise one slowly. In deep
learning the first is what matters, which is part of why the trade is usually worth taking —
and why learning-rate decay schedules, which shrink the bound over time, are near-universal
in practice.

---

## 10. When Adam fails to converge

Adam's original convergence proof is wrong. Reddi, Kale & Kumar (2018) found the error and
gave an explicit counterexample: a **convex** online problem on which Adam converges to the
worst point in the feasible set.

The problem: $x \in [-1, 1]$, and at each step the loss is $Cx$ when $t \equiv 1 \pmod 3$
and $-x$ otherwise, with $C > 2$. The average gradient is $(C-2)/3 > 0$, so the minimizer is
$x^* = -1$. The large positive gradient arrives rarely; the small negative ones arrive twice
as often. Adam's denominator averages the squared gradients, so it *divides down* the rare
large gradient relative to the frequent small ones, and the frequent direction wins.

**Experiment (measured).** $C = 3$, $\beta_1 = 0$, $\beta_2 = 1/(1+C^2) = 0.1$, step size
$0.1/\sqrt{t}$, 6000 steps, with projection back into $[-1,1]$:

| optimizer | converges to | final average regret $R_T/T$ |
|---|---|---|
| Adam | $x = +1.0$ (the **worst** feasible point) | 0.6517 |
| AMSGrad | $x = -0.9991$ | 0.029 |
| SGD | $x = -0.9974$ | 0.0001 |

![Reddi counterexample](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/08_reddi_counterexample.png)

*(a) Adam walks confidently to the wrong end of the interval and stays there — the mean of
its last 1000 iterates is 0.9994. (b) Average regret, which must go to zero for a
no-regret algorithm. Adam's rises and settles at 0.6517. AMSGrad fixes this by keeping
$\hat v_t^{\max} = \max(\hat v_{t-1}^{\max}, \hat v_t)$, so the denominator can never
shrink and a rare large gradient can never be forgotten.*

Two honest caveats. This construction needs $\beta_2$ chosen adversarially against $C$; at
the default $\beta_2 = 0.999$ the failure requires a correspondingly extreme problem. And
AMSGrad, despite fixing the proof, has not displaced Adam in practice — its monotone
denominator makes it conservative, and the empirical gains are small or absent. The
practical takeaway is not "use AMSGrad" but "Adam has no convergence guarantee, so an
unexplained training failure is allowed to be the optimizer's fault."

The other well-documented weakness is generalization: Adam often reaches a lower *training*
loss than SGD with momentum while generalizing slightly worse, particularly on image
classification (Wilson et al., 2017). Part of that gap was later traced to Adam's
interaction with L2 regularization — because L2 adds $\lambda w$ to the gradient, it also
enters the second moment and gets rescaled per parameter, which is not what weight decay is
supposed to do. Decoupling them gives **AdamW** (Loshchilov & Hutter, 2019), now the default
for transformer training. These notes do not test the generalization claim; it needs a real
benchmark with a held-out set, not a toy problem.

---

## 11. A real network, end to end

A from-scratch NumPy MLP (2–64–64–3, ReLU hidden, softmax output, cross-entropy) on a
three-class spiral with 1500 points, mini-batch 64, 80 epochs, 5 seeds. Learning rate tuned
per optimizer over an 18-point grid.

| optimizer | tuned $\eta$ | median epochs to 90% train acc | final loss (mean ± sd) | final acc |
|---|---|---|---|---|
| SGD | 0.4865 | 9 | 0.0148 ± 0.0034 | 99.4% |
| Momentum | 0.0235 | 7 | 0.0125 ± 0.0004 | 99.7% |
| RMSProp | 0.0070 | **4** | 0.0117 ± 0.0025 | 99.7% |
| Adam | 0.0128 | 5 | **0.0086 ± 0.0047** | 99.7% |

![Spiral MLP training](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/10_mlp_spiral.png)

*Training loss and accuracy, mean ± one standard deviation over 5 seeds, each optimizer at
its own tuned learning rate. Adam and RMSProp reach 90% accuracy in roughly half the epochs
SGD needs, and Adam ends at the lowest loss. Two details worth noticing: the gap is about
2×, not the order of magnitude that optimizer comparisons at a **shared** learning rate tend
to suggest (the same inflation examined in
[momentum-optimization.md](momentum-optimization.md)); and SGD's grey band is much wider
throughout, with occasional loss spikes that the adaptive methods do not show — that
variance across seeds is arguably the more practically relevant difference than the speed.*

![Summary of measured results](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/12_summary_table.png)

*Every row is a number measured by the code in this document. The pattern across all of them:
Adam is rarely the fastest at anything, and rarely bad at anything — which is exactly the
profile of a good default.*

---

## 12. Implementation from scratch

The optimizer, matching `torch.optim.Adam`'s documented algorithm:

```python
import numpy as np

class Adam:
    """m_t = b1*m + (1-b1)*g ;  v_t = b2*v + (1-b2)*g^2
       w  -= lr * (m/(1-b1^t)) / (sqrt(v/(1-b2^t)) + eps)
       b2=0 with no division recovers momentum; b1=0 recovers RMSProp."""

    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g ** 2
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params
```

The network it was tested on (the full training loop is in `code/partD.py`):

```python
def forward(params, X):
    cache, h = [X], X
    n = len(params) // 2
    for l in range(n):
        z = h @ params[2 * l] + params[2 * l + 1]
        if l < n - 1:
            h = np.maximum(z, 0)                      # ReLU
        else:
            z = z - z.max(1, keepdims=True)           # stable softmax
            e = np.exp(z)
            h = e / e.sum(1, keepdims=True)
        cache.append(h)
    return h, cache

def backward(params, cache, y):
    n = len(params) // 2
    grads = [None] * len(params)
    delta = cache[-1].copy()
    delta[np.arange(len(y)), y] -= 1                  # softmax + cross-entropy
    delta /= len(y)
    for l in reversed(range(n)):
        grads[2 * l] = cache[l].T @ delta
        grads[2 * l + 1] = delta.sum(0, keepdims=True)
        if l > 0:
            delta = (delta @ params[2 * l].T) * (cache[l] > 0)
    return grads
```

The framework equivalents:

```python
# PyTorch
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)  # decoupled

# Keras
optimizer = keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999)
```

PyTorch is not installed in the environment these notes were generated in, so the framework
snippets are unrun; the NumPy implementation above was checked against the algorithm block
in the PyTorch documentation line by line. One real difference to be aware of: PyTorch
applies $\epsilon$ **outside** the square root, as written above, while some
implementations add it inside ($\sqrt{\hat v + \epsilon}$). The two differ measurably when
$\hat v$ is very small, which is exactly the regime where a parameter has been receiving no
gradient signal.

---

## 13. Practical guidance

- **Start with Adam at $\eta = 10^{-3}$** and the default betas. It is the best first guess
  for most problems, and the measurements above are consistent with that — not because it
  wins, but because it is never badly wrong.
- **Tune $\eta$ first, and tune it on a log grid.** Every comparison in these notes changed
  character once each method got its own learning rate.
- **Use a decay schedule.** Adam's step is bounded by roughly $\eta$
  ([section 9](#9-the-bounded-step-and-why-adam-is-slow-to-high-precision)), so the final
  precision is capped by the final learning rate. Cosine or linear decay to near zero is the
  standard choice.
- **Use warmup for transformers and other deep stacks.** Bias correction fixes the average,
  but the denominator is still estimated from very few samples in the first few hundred
  steps, which is why $\hat v$ can be wildly wrong early even with correction applied.
- **Use AdamW, not Adam + L2**, when you want weight decay. This is the single most common
  practical fix.
- **Do not assume the speedup transfers.** Adam's advantage depends on your loss surface
  being roughly axis-aligned in your parameterization
  ([section 5](#5-what-adaptivity-actually-buys-the-rotation-test)). Where it is not, Adam
  is an ordinary optimizer with two buffers per parameter.
- **Consider SGD with momentum for image classification** and anywhere a well-tuned schedule
  is affordable; the generalization gap reported by Wilson et al. is still the main reason
  practitioners keep it.
- **Budget the memory.** Adam stores two extra buffers per parameter, so optimizer state is
  roughly 2× the model size in the same precision — the usual reason a model that fits for
  inference does not fit for training.

---

## 14. Key takeaways

1. Adam is momentum (first moment, direction) divided by RMSProp (second moment,
   per-parameter scale), plus bias correction. Each of those three pieces is testable
   separately and was tested separately here.
2. Bias correction is not a gentle warm-up. Without it, the step is too **large**, peaking
   at **6.569×** the corrected size at step 12 under a constant gradient, because
   $\sqrt{v}$ is biased down harder than $m$ is.
3. Adaptive scaling is not rotation invariant, and that is where its advantage comes from.
   On identical eigenvalues, rotating the basis cost RMSProp **1424×** and AdaGrad **47×**,
   while SGD, momentum and NAG were unaffected (**1.0×**).
4. AdaGrad's accumulator can only grow. Under persistent gradient noise its effective step
   decayed as $t^{-0.485}$ and its loss stalled four orders of magnitude above RMSProp's.
   RMSProp's exponential average is the whole fix.
5. The defaults are a prior for noisy gradients, not a law. On a deterministic quadratic,
   the $\beta$ grid spanned **26.7×** with $\eta$ already tuned everywhere, and
   $\beta_2 = 0.999$ was six times worse than $\beta_2 = 0.9$.
6. On sparse features the classic claim held: adaptive methods reached the loss threshold in
   **160** steps against SGD's **820**, and halved the rare-feature error at step 500. But
   all methods converged to the same place given enough steps — it is a speed claim only.
7. Adam's per-step displacement is bounded by $\eta(1-\beta_1)/\sqrt{1-\beta_2} = 3.162\eta$;
   measured median was $1.001\eta$ and maximum $1.871\eta$. This is why it is robust to $\eta$
   and also why it took 1230 steps to high precision where momentum took 127 — at a loose
   tolerance on the same problem it took 34 against momentum's 40.
8. Adam has no convergence guarantee. On Reddi et al.'s convex counterexample it converged to
   $x = +1$ when the optimum was $x = -1$, with average regret settling at 0.6517 instead of
   going to zero.
9. On a real network, tuned per optimizer, Adam reached 90% accuracy in 5 epochs against
   SGD's 9 — a 2× gain, not an order of magnitude, and the seed-to-seed variance was the
   more visible difference.

---

## 15. Further reading

- Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR. — The
  original paper, including the bias-correction derivation and the step-size bound used in
  [section 9](#9-the-bounded-step-and-why-adam-is-slow-to-high-precision).
- Reddi, S. J., Kale, S., & Kumar, S. (2018). *On the Convergence of Adam and Beyond.* ICLR.
  — The flaw in Adam's convergence proof, the counterexample reproduced in
  [section 10](#10-when-adam-fails-to-converge), and AMSGrad.
- Duchi, J., Hazan, E., & Singer, Y. (2011). *Adaptive Subgradient Methods for Online
  Learning and Stochastic Optimization.* JMLR, 12, 2121–2159. — AdaGrad.
- Tieleman, T., & Hinton, G. (2012). *Lecture 6.5 — RMSProp.* COURSERA: Neural Networks for
  Machine Learning. — RMSProp, never published as a paper.
- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR. —
  AdamW, and why Adam + L2 is not weight decay.
- Wilson, A. C., Roelofs, R., Stern, M., Srebro, N., & Recht, B. (2017). *The Marginal Value
  of Adaptive Gradient Methods in Machine Learning.* NeurIPS. — The generalization gap.
- Dozat, T. (2016). *Incorporating Nesterov Momentum into Adam.* ICLR Workshop. — NAdam.
- Zhang, J., Karimireddy, S. P., et al. (2020). *Why are Adaptive Methods Good for Attention
  Models?* NeurIPS. — Heavy-tailed gradient noise as an explanation for Adam's dominance in
  language models.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, chapter 8. MIT Press.
