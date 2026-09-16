# Momentum Optimization

Momentum is the smallest useful change you can make to gradient descent: keep a running
average of past gradients and step along that average instead of along the raw gradient.
One extra line of code, one extra buffer per parameter. This document works through why
that line helps, exactly when it helps, and — importantly — when it does nothing at all.
Every figure below was produced by running code, and every number quoted in the prose is
output from an experiment that is included in full at the end. Where an experiment
contradicted the usual story, the contradiction is reported rather than smoothed over;
two of them did.

Prerequisites: [gradient-descent.md](gradient-descent.md) for the basic update rule and
[sgd.md](sgd.md) for batch/mini-batch/stochastic variants. This document feeds into
[nesterov-accelerated-gradient.md](nesterov-accelerated-gradient.md) and
[adam.md](adam.md), both of which reuse the moving-average machinery built here.

---

## Table of Contents

1. [Seeing the loss landscape](#1-seeing-the-loss-landscape)
2. [Convex vs non-convex, and the three failure modes](#2-convex-vs-non-convex-and-the-three-failure-modes)
3. [Why plain gradient descent zig-zags](#3-why-plain-gradient-descent-zig-zags)
4. [Exponentially weighted moving averages](#4-exponentially-weighted-moving-averages)
5. [The momentum update rule](#5-the-momentum-update-rule)
6. [Why momentum accelerates: rates and measurements](#6-why-momentum-accelerates-rates-and-measurements)
7. [The role of beta](#7-the-role-of-beta)
8. [Local minima and saddle points](#8-local-minima-and-saddle-points)
9. [Limitations: overshoot, oscillation, and one myth](#9-limitations-overshoot-oscillation-and-one-myth)
10. [A real network, end to end](#10-a-real-network-end-to-end)
11. [Implementation from scratch](#11-implementation-from-scratch)
12. [Practical guidance](#12-practical-guidance)
13. [Key takeaways](#13-key-takeaways)
14. [Further reading](#14-further-reading)
15. [Reproducing these results](#15-reproducing-these-results)

---

## 1. Seeing the loss landscape

Optimization is the search for parameters $w$ that minimize a loss $L(w)$. For a network
with millions of parameters that surface is unvisualizable, so everything below uses two
parameters, where the same surface can be drawn three ways:

- a **1D slice**: fix all parameters but one and plot $L$ against it;
- a **3D surface**: height above the $(w_1, w_2)$ plane is the loss;
- a **contour plot**: the same surface seen from directly above, with lines joining points
  of equal loss — the exact convention used on topographic maps.

![Three views of one loss surface](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_loss_landscape_views.png)

*The identical function $L(w_1,w_2)=\frac{1}{2}(w_1^2+4w_2^2)$ rendered as 1D slices, a 3D
surface, and a contour map. The two slices in (a) have different steepness, which is the
whole story of this document in miniature: the loss is four times more curved along $w_2$
than along $w_1$. In (b) that shows up as a valley rather than a symmetric bowl, and in
(c) as elliptical rather than circular contours. Contour plots are the standard tool for
comparing optimizers because an optimizer's path can be drawn directly on top of them, and
because contour spacing encodes gradient magnitude: tightly packed lines mean a steep
region, widely spaced lines mean a flat one.*

Two properties of the gradient $\nabla_w L$ carry over unchanged to high dimensions and are
worth fixing now: it points in the direction of **steepest ascent**, so descent moves along
$-\nabla_w L$; and it is **perpendicular to the contour line** through the current point.
The second property is what makes zig-zagging inevitable in a narrow valley, as
[section 3](#3-why-plain-gradient-descent-zig-zags) shows.

---

## 2. Convex vs non-convex, and the three failure modes

A function is **convex** if the line segment between any two points on its graph lies on or
above the graph:

$$L(\alpha w_1 + (1-\alpha) w_2) \le \alpha L(w_1) + (1-\alpha) L(w_2), \quad \forall \alpha \in [0,1]$$

Equivalently, for twice-differentiable $L$, the Hessian $\nabla^2 L$ is positive
semi-definite everywhere. Convexity buys one enormous guarantee: **any local minimum is a
global minimum**, so gradient descent from any starting point converges to the same
solution. Linear regression with squared error and logistic regression with cross-entropy
are convex. Essentially no neural network is.

![Convex vs non-convex loss](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_convex_vs_nonconvex.png)

*Left: a convex loss — one basin, and every downhill path ends at the same point, so the
starting position is irrelevant. Right: the non-convex function
$L(w)=0.1w^2+\sin(w)+0.6\sin(4.3w)$ used for the experiment in
[section 8](#8-local-minima-and-saddle-points), which has eight minima over $[-6,6]$ — seven
local traps (purple) and one global minimum (star). Here the initialization decides the outcome, which is
why weight initialization schemes and optimizer choice matter in a way they simply do not
for convex problems.*

Non-convexity produces three distinct difficulties, and they call for different responses:

1. **Local minima.** The gradient is zero and every direction goes uphill, but the loss is
   higher than the global minimum. In high dimensions these are rarer than intuition
   suggests: a critical point is a local minimum only if *all* $d$ Hessian eigenvalues are
   positive, which becomes exponentially unlikely as $d$ grows (Dauphin et al., 2014).
2. **Saddle points.** The gradient is zero, but the surface curves up in some directions and
   down in others. These dominate in high dimensions, and the problem they cause is not
   getting permanently stuck — it is spending an enormous number of steps crawling across a
   near-flat region where $\|\nabla L\| \approx 0$.
3. **High curvature / ill-conditioning.** No critical point is involved at all. The surface
   is simply much more curved in some directions than others, which bounds the usable
   learning rate to what the *steepest* direction tolerates, while progress is limited by
   the *flattest* one.

![Saddle point and ill-conditioned valley](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_saddle_and_curvature.png)

*Left: $L=w_1^2-w_2^2$, the canonical saddle. The origin is a critical point but not a
minimum — the surface falls away along $w_2$. Right: an ill-conditioned bowl with condition
number $\kappa=20$, drawn as contours. The ellipses are stretched because curvature along
$w_2$ is twenty times that along $w_1$. The optimizer must take steps small enough not to
diverge along the steep axis while needing many of them to cross the flat axis; this
tension, not local minima, is the practical bottleneck in most real training runs.*

The quantity that governs difficulty in case 3 is the **condition number** of the Hessian:

$$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$$

Momentum's most reliable benefit is a reduction of the $\kappa$ dependence of the
convergence rate, which is quantified in [section 6](#6-why-momentum-accelerates-rates-and-measurements).

---

## 3. Why plain gradient descent zig-zags

Take the quadratic $L(w) = \frac{1}{2}(w_1^2 + 100\,w_2^2)$, so $\kappa = 100$. The gradient
descent update decouples across the two axes:

$$w_i^{(t)} = (1 - \eta\,\lambda_i)\, w_i^{(t-1)} \quad \Rightarrow \quad w_i^{(t)} = (1-\eta\lambda_i)^t\, w_i^{(0)}$$

Each coordinate decays geometrically with its own factor $|1-\eta\lambda_i|$, and two facts
follow immediately:

- **Stability** requires $|1-\eta\lambda_i| < 1$ for every $i$, i.e. $\eta < 2/\lambda_{\max}$.
  The steepest direction alone caps the learning rate.
- **Speed** is set by the slowest coordinate, $|1-\eta\lambda_{\min}|$, which is close to 1
  when $\lambda_{\min} \ll \lambda_{\max}$.

Optimizing the trade-off gives $\eta^\star = \frac{2}{\lambda_{\max}+\lambda_{\min}}$ and a
best-case per-step contraction of

$$\rho_{\text{GD}} = \frac{\kappa - 1}{\kappa + 1}$$

which for $\kappa = 100$ is $0.9802$ — a 2% improvement per step, no matter how carefully
the learning rate is tuned.

**Experiment (measured).** Starting from $w^{(0)}=(-10, 1)$, a learning rate grid search over
400 log-spaced values, requiring $\|w\| < 10^{-3}\|w^{(0)}\|$ *and staying there*:

| optimizer | best $\eta$ found | steps to tolerance |
|---|---|---|
| gradient descent | 0.01976 | **346** |
| momentum, $\beta=0.9$ | 0.00472 | **113** |

The measured 346 steps match the theory closely: $\ln(10^{-3}) / \ln(0.9802) = 345.4$. The
best learning rate found by search, 0.01976, matches $2/(\lambda_{\max}+\lambda_{\min}) = 2/101 = 0.0198$.

![Zig-zag versus momentum](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_zigzag_vs_momentum.png)

*(a) The first 120 steps of each method on the $\kappa=100$ bowl, each at its own best-tuned
learning rate. Gradient descent (orange) oscillates across the narrow axis on every single
step while creeping along the flat axis — the classic zig-zag, and it is not a tuning
mistake: at any smaller $\eta$ the horizontal progress is even slower. Momentum (blue)
oscillates too, but at a far lower frequency, because the alternating vertical components
of successive gradients partially cancel inside the velocity buffer while the consistently
horizontal components accumulate. (b) Loss against step on a log scale. Both are straight
lines, confirming geometric convergence; momentum's line is roughly three times steeper. A
detail worth noticing: the "staying there" requirement matters, because an oscillating
trajectory can dip under any tolerance for a single lucky step. Counting those dips makes
$\beta$ comparisons non-monotonic and meaningless — an earlier version of this experiment
did exactly that and reported $\beta=0.8$ converging faster than the theoretical optimum.*

The measured speedup here is **3.06×**.

---

## 4. Exponentially weighted moving averages

Momentum is built from one primitive, the exponentially weighted moving average (EWMA). For
a stream $x_1, x_2, \dots$:

$$v_t = \beta\, v_{t-1} + (1-\beta)\, x_t, \qquad v_0 = 0$$

Unrolling the recursion shows what $v_t$ actually is:

$$v_t = (1-\beta)\sum_{k=1}^{t} \beta^{\,t-k} x_k$$

— a weighted sum of the entire history, with exponentially decaying weights. The weights sum
to $1-\beta^t \to 1$, so $v_t$ is a proper average. (This identity was checked numerically:
the recursive and closed-form values agree to $2.2 \times 10^{-16}$ over 60 steps, i.e. to
floating-point precision.)

Since $\beta^{\,1/(1-\beta)} \approx e^{-1}$, a term's weight has decayed by a factor of $e$
after about

$$\text{effective window} \approx \frac{1}{1-\beta} \text{ steps}$$

so $\beta=0.9$ averages roughly the last 10 gradients and $\beta=0.99$ roughly the last 100.

If the $x_t$ are independent with standard deviation $\sigma$, the stationary standard
deviation of $v_t$ is

$$\sigma_v = \sigma \sqrt{\frac{1-\beta}{1+\beta}}$$

**Experiment (measured).** A sinusoid corrupted with Gaussian noise of measured standard
deviation 0.5718, smoothed at three values of $\beta$, with the residual measured after
correcting for the lag each filter introduces:

| $\beta$ | window $1/(1-\beta)$ | measured lag | residual noise (measured) | $\sigma\sqrt{(1-\beta)/(1+\beta)}$ (theory) |
|---|---|---|---|---|
| 0.5 | 2 | 1 step | 0.327 | 0.330 |
| 0.9 | 10 | 9 steps | 0.130 | 0.131 |
| 0.98 | 50 | 25 steps | 0.420 | 0.058 |

The first two rows match theory to within 1%. The third does not, and the reason is
instructive rather than a bug: with an effective window of 50 steps against an underlying
signal whose period is about 138 steps, the filter is no longer averaging noise around a
locally constant value — it is averaging away the signal itself. Lag correction cannot
repair that distortion, so the measured residual is dominated by signal error, not noise.
This is the smoothing/staleness trade-off that reappears as the cost of large $\beta$ in
[section 7](#7-the-role-of-beta).

![EWMA smoothing](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_ewma_smoothing.png)

*(a) The raw noisy stream (grey), the true underlying signal (dashed), and EWMA outputs at
three decay factors. $\beta=0.5$ tracks the signal responsively but stays noisy; $\beta=0.9$
is a good compromise; $\beta=0.98$ is beautifully smooth and visibly behind — its peaks
arrive late and are flattened, which is exactly what "using a stale gradient" looks like.
(b) The trade-off quantified: the effective window grows as $1/(1-\beta)$, the measured lag
grows with it, and the residual noise falls and then rises again once distortion takes over.*

Note that $v_0 = 0$ biases early estimates toward zero. The bias-corrected form
$\hat{v}_t = v_t / (1-\beta^t)$ fixes it; classical momentum simply ignores the issue,
because the first few steps of training rarely matter. Adam does correct for it — see
[adam.md](adam.md).

---

## 5. The momentum update rule

Two conventions appear in the literature and in framework source code. Both are correct;
confusing them is the single most common source of reproduction failures.

**Classical / heavy-ball form** (what `torch.optim.SGD` implements):

$$v_t = \beta\, v_{t-1} + \nabla_w L(w_{t-1}), \qquad w_t = w_{t-1} - \eta\, v_t$$

**EWMA form** (how the idea is usually taught, and what Adam uses):

$$v_t = \beta\, v_{t-1} + (1-\beta)\nabla_w L(w_{t-1}), \qquad w_t = w_{t-1} - \eta'\, v_t$$

The two generate **identical trajectories** when $\eta' = \eta/(1-\beta)$. This was verified
rather than asserted: over 80 steps on a 2D quadratic the two parameter sequences agree to a
maximum absolute difference of $3.9\times10^{-16}$. The practical consequence is that
raising $\beta$ in the classical form silently raises the effective step size, because
unrolling the velocity under a constant gradient $g$ gives

$$v_\infty = g \sum_{k=0}^{\infty}\beta^k = \frac{g}{1-\beta} \quad\Rightarrow\quad \eta_{\text{eff}} = \frac{\eta}{1-\beta}$$

Measured directly: with $g=1$ and $\beta=0.9$ the velocity converges to exactly 10.0000; with
$\beta=0.99$ it reaches 99.343 after 500 steps, en route to 100. So switching
$\beta: 0 \to 0.9$ at fixed $\eta$ multiplies the asymptotic step by ten. **A large fraction
of the benefit people attribute to momentum is really this hidden learning-rate increase**,
which is why [section 10](#10-a-real-network-end-to-end) runs the control experiment.

![The momentum update step](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_momentum_update_diagram.png)

*The dataflow of one momentum step. The gradient is computed exactly as in SGD; the only
structural change is the velocity buffer, which mixes the new gradient with the decayed
memory of every previous one and is then carried forward to the next iteration. Setting
$\beta=0$ collapses the blue box to $v_t = g_t$ and recovers plain gradient descent
exactly — verified in code: with $\beta=0$ the momentum optimizer and a hand-written
gradient-descent loop produce bitwise identical parameters (max absolute difference 0.0)
after 20 steps on a small network.*

The physical reading is the one from the lecture: $w$ is a ball rolling on the loss surface,
$-\nabla L$ is the force of gravity along the slope, $v$ is velocity, and $\beta$ is a
friction/drag coefficient. With $\beta=0$ the ball has no inertia and its velocity is
re-derived from scratch at every instant; with $\beta$ near 1 it barely loses speed and
carries through flat stretches and small bumps. The analogy is genuinely useful but has a
limit worth stating: in real physics the ball's kinetic energy causes overshoot *and* it
eventually dissipates; here $\beta$ is a fixed constant, so the "friction" never adapts, and
the overshoot measured in [section 9](#9-limitations-overshoot-oscillation-and-one-myth) does
not shrink as the ball approaches the minimum.

---

## 6. Why momentum accelerates: rates and measurements

On a quadratic with Hessian eigenvalue $\lambda$, the classical momentum update is a
second-order linear recurrence:

$$w^{(t+1)} = (1 + \beta - \eta\lambda)\,w^{(t)} - \beta\,w^{(t-1)}$$

Its convergence rate is the larger root modulus of the characteristic polynomial
$z^2 - (1+\beta-\eta\lambda)z + \beta = 0$. When the discriminant is negative the roots are
complex conjugates with product $\beta$, so both have modulus $\sqrt{\beta}$:

$$\rho_{\text{momentum}} = \sqrt{\beta} \quad \text{(independent of } \lambda \text{)}$$

This is the whole mechanism. In the complex-root regime every eigendirection contracts at
the *same* rate $\sqrt{\beta}$, so the spread of curvatures stops mattering. Optimizing over
both hyperparameters gives (Polyak, 1964):

$$\beta^\star = \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^2, \qquad \eta^\star = \frac{4}{(\sqrt{\lambda_{\max}}+\sqrt{\lambda_{\min}})^2}, \qquad \rho^\star = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$$

Compare with gradient descent's $\frac{\kappa-1}{\kappa+1}$: the number of steps to a fixed
accuracy improves from $O(\kappa)$ to $O(\sqrt{\kappa})$. For $\kappa = 100$ that is the
difference between $\rho = 0.9802$ and $\rho = 0.8182$.

**Experiment (measured).** Sweeping $\beta$ over 40 values on $[0, 0.97]$ and re-tuning
$\eta$ by grid search at each one, the empirical optimum was $\beta = 0.696$, reaching
tolerance in **35 steps** versus 346 for gradient descent — a **9.9× speedup**. Theory
predicts $\beta^\star = \left(\frac{10-1}{10+1}\right)^2 = 0.669$. Measured 0.696 against
predicted 0.669 is agreement to within the resolution of a 40-point grid, which is about as
close as this experiment can resolve.

---

## 7. The role of beta

$\beta$ controls how much of the past survives into the present step. The two ways of asking
"what does $\beta$ cost me?" give different-looking answers, so both are measured here.

![Beta sweep](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_beta_sweep.png)

*(a) Loss curves with $\eta$ re-tuned separately for each $\beta$; all are straight lines on
a log axis, and the slope steepens up to $\beta \approx 0.7$ and then flattens again.
(b) The less forgiving view: every $\beta$ forced to use the learning rate that is optimal
for plain gradient descent, $\eta = 0.0198$. Nothing diverges here, but the U-shape is sharp —
$\beta=0.8$ needs 54 steps while $\beta=0.99$ needs 1363, four times worse than using no
momentum at all. (c) The fine sweep with $\eta$ re-tuned at each of 40 values of $\beta$,
with the theoretical optimum marked. The curve is a clean U with a narrow floor: the cost of
$\beta$ being somewhat too small is mild, the cost of it being too large is severe.*

With $\eta$ tuned per $\beta$:

| $\beta$ | best $\eta$ | steps |
|---|---|---|
| 0.0 | 0.01976 | 346 |
| 0.5 | 0.02925 | 108 |
| 0.67 | 0.03283 | 46 |
| 0.8 | 0.01334 | 52 |
| 0.9 | 0.00472 | 113 |
| 0.95 | 0.00076 | 231 |
| 0.99 | 0.00011 | 1221 |

Read the middle column as carefully as the right one. As $\beta$ grows past the optimum, the
best learning rate collapses — by $\beta = 0.99$ it is 180× smaller than gradient descent's.
The velocity buffer is doing the stepping, so $\eta$ must shrink to compensate, and the two
effects nearly cancel. **$\beta$ and $\eta$ are not independent knobs**; changing one without
the other changes the effective step size more than it changes the smoothing.

Why each extreme fails:

- **$\beta$ too small** — barely any history is averaged, so oscillating gradient components
  never get the chance to cancel. At $\beta = 0$ this is exactly gradient descent, with the
  full $O(\kappa)$ step count.
- **$\beta$ too large** — the velocity is dominated by gradients measured many steps ago, at
  parameter values that no longer describe where the optimizer is. This is the same lag that
  flattened the $\beta = 0.98$ curve in [section 4](#4-exponentially-weighted-moving-averages);
  here it shows up as sluggish response to curvature changes and long-lived oscillations.

In practice $\beta \in [0.9, 0.99]$ is the standard range, with 0.9 the near-universal
default. Note the tension with the measurement above, where the optimum was 0.696: for a
*single* well-understood quadratic, a smaller $\beta$ is better. The defaults are larger
because real loss surfaces have gradient noise that also needs averaging, and because
$\kappa$ is typically far larger than 100, which pushes $\beta^\star$ upward
($\kappa = 10^4$ gives $\beta^\star = 0.96$).

---

## 8. Local minima and saddle points

**Local minima.** The physical intuition says a ball with enough speed rolls straight through
a shallow dip. That is real, but it is worth measuring how often it actually rescues a run.
Using the non-convex function from [section 2](#2-convex-vs-non-convex-and-the-three-failure-modes),
221 starting points evenly spaced over $[-5.5, 5.5]$ were run with both optimizers at
identical $\eta = 0.02$, $\beta = 0.9$, for 400 steps:

- gradient descent reached the global minimum from **32** of 221 starts (14.5%);
- momentum reached it from **41** of 221 starts (18.6%);
- there were **9** starts (4.1%) where momentum escaped a trap and gradient descent did not;
- there were **0** starts where the reverse happened.

So momentum never did worse, but it converted only about one start in 25. The
escape-from-local-minima story is true and is *not* where most of momentum's practical value
comes from.

![Escaping a local minimum](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/08_local_minimum_escape.png)

*One of the nine starts where the two methods diverge, $w_0 = -3.55$. Gradient descent
(orange dots) settles into the first basin it meets and ends at $w = -3.129$ with
$L = 0.5009$; momentum (blue) carries through that basin, overshoots the global minimum,
oscillates, and settles at $w = -1.776$ with $L = -1.2495$. Panel (b) shows the flat orange
line of a run that stopped improving at step ~10 against momentum's noisy descent — and the
oscillation visible in the blue curve up to step 60 is the same overshoot quantified in
[section 9](#9-limitations-overshoot-oscillation-and-one-myth).*

**Saddle points.** This is where momentum's help is much more consistent. At a saddle the
gradient along the escape direction is tiny but *consistently signed*, which is precisely the
condition under which an exponential average amplifies rather than cancels. On
$L = w_1^2 - w_2^2$ starting at $(1, 10^{-3})$ with $\eta = 0.01$:

| optimizer | steps until $\lvert w_2\rvert > 0.5$ |
|---|---|
| gradient descent | **314** |
| momentum, $\beta = 0.9$ | **65** |

A **4.83×** speedup, and unlike the local-minimum result it does not depend on a lucky
initialization.

![Saddle escape](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/09_saddle_escape.png)

*(a) Both trajectories on the saddle's contours. Both eventually escape — the saddle is not a
trap for either method — but the time spent loitering near the origin differs by nearly a
factor of five. (b) $\lvert w_2 \rvert$ on a log axis, where both curves are straight lines,
confirming that escape is exponential growth along the negative-curvature direction. Momentum
simply has the larger growth exponent, because the small, consistently-signed gradient along
$w_2$ accumulates in the velocity buffer up to the $1/(1-\beta) = 10\times$ amplification
derived in [section 5](#5-the-momentum-update-rule).*

---

## 9. Limitations: overshoot, oscillation, and one myth

**Overshoot is real and measurable.** On the simplest possible problem,
$L = \frac{1}{2}w^2$ from $w_0 = 1$ with $\eta = 0.1$:

| | gradient descent | momentum, $\beta=0.9$ |
|---|---|---|
| most extreme point past the minimum | never crosses zero | $w = -0.6037$ (**60.4%** of the initial distance, on the far side) |
| sign changes before settling | 0 | **12** |
| steps until $\lvert w\rvert < 0.01$ and stays there | **44** | **81** |
| $\lvert w \rvert$ at step 120 | $3.2\times10^{-6}$ | $1.1\times10^{-3}$ |

![Overshoot and oscillation](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/10_overshoot_oscillation.png)

*(a) Gradient descent approaches the minimum monotonically; momentum shoots 60% of the way
past it, comes back, overshoots again, and rings down over roughly 12 crossings. The shaded
region is the excursion on the wrong side of the minimum. (b) The same runs on a log axis,
and the panel that corrects a plausible-sounding claim: on a single well-conditioned
direction momentum is not just bouncier, it is **slower**. Gradient descent contracts by
$|1-\eta\lambda| = 0.9$ per step; momentum's complex roots give $\sqrt{\beta} = 0.9487$. By
step 120 that gap is nearly three orders of magnitude.*

This deserves emphasis because it is easy to get backwards. Momentum does not make every
problem faster. It makes the rate **uniform across eigendirections** at $\sqrt{\beta}$,
which is a huge win when $\kappa$ is large and a genuine loss when $\kappa \approx 1$, since
gradient descent can then simply set $\eta \approx 1/\lambda$ and converge in a handful of
steps. An early draft of this document claimed momentum reached the tolerance first here; the
claim came from counting the first step below the threshold, which momentum hits transiently
while oscillating through zero. Requiring the trajectory to *stay* below threshold reverses
the result.

**The myth: momentum averages away gradient noise.** The smoothing picture in
[section 4](#4-exponentially-weighted-moving-averages) makes it tempting to conclude that
momentum reduces the noise floor of stochastic optimization. Tested directly on a 20D
quadratic with Gaussian gradient noise ($\sigma = 1$), 200 seeds, 400 steps, comparing the
final loss averaged over the last 50 steps:

| arm | final loss (mean ± sd) | cos between consecutive updates |
|---|---|---|
| SGD, $\eta = 0.02$ | 0.1028 ± 0.0235 | −0.02 |
| momentum $\beta=0.9$, **matched** effective step | 0.0996 ± 0.0268 | **+0.90** |
| momentum $\beta=0.9$, same $\eta$ (10× effective step) | 1.010 ± 0.148 | +0.87 |

At matched effective step size the noise floors are statistically indistinguishable — a 3%
difference against a 23% seed-to-seed standard deviation. Momentum does *not* lower the stationary noise
floor. What it changes is the **shape of the path**: the cosine similarity between
consecutive update vectors goes from −0.02 (each step nearly uncorrelated with, and slightly
opposed to, the last) to +0.90 (a smooth, persistent direction). Meanwhile the third row
shows what happens when the effective step size is *not* matched: the noise floor is 10×
worse, exactly as the $\eta_{\text{eff}} = \eta/(1-\beta)$ analysis predicts. Reports that
"momentum increased my final loss" are usually this, and the fix is to divide $\eta$ by
$1/(1-\beta)$ when turning momentum on.

![Noisy gradients](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/11_noisy_gradients.png)

*(a) Loss curves under gradient noise, mean ± one standard deviation over 200 seeds. The SGD
and matched-step momentum bands overlap almost completely; the same-$\eta$ momentum arm
plateaus an order of magnitude higher. (b) Cosine similarity between consecutive updates —
the quantity that actually changes. Momentum's smoothing is visible in the trajectory, not in
the noise floor.*

Momentum also carries two ordinary costs: one extra buffer per parameter (a 2× increase in
optimizer state over plain SGD, though negligible beside Adam's 3×), and one more
hyperparameter to tune, which as [section 7](#7-the-role-of-beta) showed is not independent
of the learning rate.

---

## 10. A real network, end to end

A from-scratch NumPy MLP (2–32–32–1, ReLU hidden, sigmoid output, binary cross-entropy) on a
1000-point two-moons dataset, mini-batch size 64, 200 epochs, 5 seeds. Three arms, because
the naive comparison and the controlled one give different answers:

| arm | median epochs to 97% train accuracy | final loss (mean) |
|---|---|---|
| SGD, $\eta = 0.05$ | **58** | 0.0097 |
| momentum $\beta = 0.9$, $\eta = 0.05$ | **8** | 0.0009 |
| SGD, $\eta = 0.5$ (matched effective step) | **7** | 0.0009 |

Against SGD at the same learning rate, momentum is **7.25× faster**. Against SGD given the
matched effective learning rate $\eta/(1-\beta) = 0.5$, momentum is **0.88×** — slightly
*slower*, and the final losses are identical.

![MLP training curves](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/12_mlp_training.png)

*Training loss and accuracy, mean ± one standard deviation over 5 seeds. The blue (momentum)
and purple (SGD at 10× learning rate) curves lie on top of each other for the entire run,
while orange (SGD at the original rate) trails badly. On this problem, momentum's entire
advantage is the effective-step-size increase — this network's loss surface is well enough
conditioned that there is nothing left for the smoothing to fix.*

That does not contradict [section 3](#3-why-plain-gradient-descent-zig-zags); it delimits it.
On the $\kappa = 100$ quadratic, both methods got their own tuned learning rate and momentum
still won by 3.06×, because there the acceleration is a genuine rate improvement from
$O(\kappa)$ to $O(\sqrt{\kappa})$. The honest summary is that **momentum's benefit scales
with the ill-conditioning of the problem**, and on an easy problem a bigger learning rate is
all you were getting. Real deep networks are strongly ill-conditioned, which is why momentum
earns its place — but that is an argument from the literature and from larger experiments,
not something this two-moons run demonstrates.

![Summary of measured results](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/13_summary_table.png)

*Every row is a number measured by the code in this document rather than quoted from
elsewhere. Read together, the pattern is: momentum wins decisively on ill-conditioning and
saddles, wins occasionally on local minima, and wins nothing on well-conditioned problems or
on the stochastic noise floor.*

---

## 11. Implementation from scratch

The optimizer itself, in the classical form used by PyTorch:

```python
import numpy as np

class Momentum:
    """Classical / heavy-ball momentum:
           v_t = beta * v_{t-1} + g_t
           w_t = w_{t-1} - lr * v_t
       beta = 0 reduces exactly to plain gradient descent."""

    def __init__(self, params, lr=0.1, beta=0.9):
        self.lr, self.beta = lr, beta
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        for i, (p, g) in enumerate(zip(params, grads)):
            self.v[i] = self.beta * self.v[i] + g
            p -= self.lr * self.v[i]          # in-place update
        return params
```

The network it was tested on, also from scratch:

```python
def init_params(sizes, seed=0):
    r = np.random.default_rng(seed)
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        params += [r.normal(0, np.sqrt(2.0 / a), (a, b)), np.zeros((1, b))]  # He init
    return params

def forward(params, X):
    cache, h = [X], X
    n_layers = len(params) // 2
    for l in range(n_layers):
        W, b = params[2 * l], params[2 * l + 1]
        z = h @ W + b
        h = np.maximum(z, 0) if l < n_layers - 1 else 1 / (1 + np.exp(-z))
        cache.append(h)
    return h, cache

def bce(yhat, y):
    eps = 1e-9
    return float(-np.mean(y * np.log(yhat + eps) + (1 - y) * np.log(1 - yhat + eps)))

def backward(params, cache, y):
    n_layers = len(params) // 2
    grads = [None] * len(params)
    delta = (cache[-1] - y) / y.shape[0]      # sigmoid + BCE collapse to this
    for l in reversed(range(n_layers)):
        grads[2 * l] = cache[l].T @ delta
        grads[2 * l + 1] = delta.sum(0, keepdims=True)
        if l > 0:
            delta = (delta @ params[2 * l].T) * (cache[l] > 0)   # ReLU derivative
    return grads

def train(beta, lr=0.05, epochs=200, batch=64, seed=0):
    X, y = two_moons(1000, seed=1)
    params = init_params([2, 32, 32, 1], seed=seed)
    opt = Momentum(params, lr=lr, beta=beta)
    r = np.random.default_rng(seed)
    losses, accs = [], []
    for _ in range(epochs):
        idx = r.permutation(len(X))
        for s in range(0, len(X), batch):
            b = idx[s:s + batch]
            yhat, cache = forward(params, X[b])
            opt.step(params, backward(params, cache, y[b]))
        yhat, _ = forward(params, X)
        losses.append(bce(yhat, y))
        accs.append(float(np.mean((yhat > 0.5) == (y > 0.5))))
    return np.array(losses), np.array(accs), params
```

Two correctness checks were run rather than assumed:

1. **$\beta=0$ must equal gradient descent.** Running `Momentum(..., beta=0.0)` alongside a
   hand-written `p -= lr * g` loop for 20 steps on identical initial parameters gives a
   maximum absolute parameter difference of **0.0** — bitwise identical.
2. **The rule must match PyTorch's.** Transcribing the algorithm block from the
   `torch.optim.SGD` documentation into NumPy and feeding both the same gradient sequence
   gives a maximum absolute difference of **0.0**. (PyTorch is not installed in the
   environment these notes were generated in, so this checks the update rule against its
   published specification, not against a live PyTorch run.)

The framework equivalents, for reference:

```python
# PyTorch
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

# Keras
optimizer = keras.optimizers.SGD(learning_rate=0.05, momentum=0.9)
```

One subtlety worth carrying into real code: PyTorch's `dampening` parameter changes the rule
to $v_t = \beta v_{t-1} + (1-\text{dampening})g_t$, so `dampening=momentum` gives the EWMA
form of [section 5](#5-the-momentum-update-rule) and therefore a $(1-\beta)$-times smaller
effective step. Keras has no dampening argument and always uses the classical form.

---

## 12. Practical guidance

- **Start at $\beta = 0.9$.** It is the default in every major framework and a reasonable
  value across a wide range of problems. Tune the learning rate first; tune $\beta$ only if
  the learning rate is already near its stability limit.
- **When switching momentum on, divide the learning rate by $1/(1-\beta)$** — otherwise you
  are changing two things at once, and the noise-floor result in
  [section 9](#9-limitations-overshoot-oscillation-and-one-myth) shows what that costs.
- **Diagnose by watching the loss curve.** Sustained oscillation with a slow downward trend
  means the effective step is too large: reduce $\eta$, or reduce $\beta$. A curve that is
  smooth but barely descending means $\beta$ is too high and the velocity is stale.
- **Gradient clipping composes badly with high $\beta$.** Clipping bounds the gradient, but
  the velocity buffer can still accumulate to $1/(1-\beta)$ times the clip threshold.
- **Momentum helps most where curvature is uneven.** If your problem is well-conditioned,
  expect little beyond what a larger learning rate would give you.
- **Prefer Nesterov momentum when it is one flag away.** It evaluates the gradient at the
  look-ahead point $w - \eta\beta v$ and typically damps overshoot at no extra cost — see
  [nesterov-accelerated-gradient.md](nesterov-accelerated-gradient.md).
- **For most modern work, reach for Adam first** (see [adam.md](adam.md)), which combines this
  first-moment average with a per-parameter second-moment scaling. Momentum remains the
  better-understood component and still wins on some vision benchmarks when paired with a
  well-designed learning-rate schedule.

---

## 13. Key Takeaways

1. Momentum replaces the raw gradient with an exponentially weighted average of past
   gradients: $v_t = \beta v_{t-1} + g_t$, $w_t = w_{t-1} - \eta v_t$. One buffer, one line.
2. Its real mechanism is not "rolling past local minima" but making the convergence rate
   **uniform across eigendirections** at $\sqrt{\beta}$, which improves the step count from
   $O(\kappa)$ to $O(\sqrt{\kappa})$. Measured on a $\kappa=100$ bowl with both methods fully
   tuned: 346 steps → 113 at $\beta=0.9$, and 35 steps at the empirically best $\beta=0.696$
   (theory predicts 0.669).
3. $\beta$ sets the effective averaging window $1/(1-\beta)$ **and** multiplies the effective
   learning rate by $1/(1-\beta)$. These two knobs are coupled; the measured best $\eta$ fell
   by 180× between $\beta=0$ and $\beta=0.99$.
4. On saddle points, momentum is reliably faster (measured 4.83×), because a small,
   consistently-signed gradient is exactly what an exponential average amplifies.
5. On local minima, the help is real but occasional: over 221 starting points it rescued 9
   (4.1%) and lost none.
6. Momentum overshoots. On $L=\frac{1}{2}w^2$ it travelled 60.4% of the initial distance past
   the minimum and crossed it 12 times, and on that well-conditioned problem it was *slower*
   to converge than plain gradient descent.
7. Momentum does **not** lower the stochastic noise floor. At matched effective step size the
   final losses were 0.1028 vs 0.0996 — a difference well inside the seed-to-seed spread.
   What it changes is path smoothness: cosine similarity between consecutive updates went
   from −0.02 to +0.90.
8. Always compare against SGD at the *matched* effective learning rate. On the two-moons MLP,
   momentum looked 7.25× faster than SGD at equal $\eta$ and 0.88× against the matched
   control.

---

## 14. Further Reading

- Polyak, B. T. (1964). *Some methods of speeding up the convergence of iteration methods.*
  USSR Computational Mathematics and Mathematical Physics, 4(5), 1–17. — The original
  heavy-ball method and the optimal $\beta^\star$, $\eta^\star$, $\rho^\star$ formulas used
  in [section 6](#6-why-momentum-accelerates-rates-and-measurements).
- Nesterov, Y. (1983). *A method for solving the convex programming problem with convergence
  rate $O(1/k^2)$.* Doklady Akademii Nauk SSSR, 269, 543–547. — The accelerated variant.
- Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). *On the importance of
  initialization and momentum in deep learning.* ICML. — The paper that established momentum
  as standard practice for deep nets, including the momentum schedule.
- Qian, N. (1999). *On the momentum term in gradient descent learning algorithms.* Neural
  Networks, 12(1), 145–151. — The physical/dynamical-systems reading of momentum.
- Goh, G. (2017). *Why Momentum Really Works.* Distill. — Interactive treatment of the
  eigenvalue analysis in [section 6](#6-why-momentum-accelerates-rates-and-measurements).
- Dauphin, Y., Pascanu, R., Gulcehre, C., Cho, K., Ganguli, S., & Bengio, Y. (2014).
  *Identifying and attacking the saddle point problem in high-dimensional non-convex
  optimization.* NeurIPS. — Why saddles, not local minima, dominate in high dimensions.
- Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR. — First
  and second moment estimates, plus the bias correction discussed in
  [section 4](#4-exponentially-weighted-moving-averages).
- Ruder, S. (2016). *An overview of gradient descent optimization algorithms.*
  arXiv:1609.04747. — Compact survey placing momentum among its relatives.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, chapter 8. MIT Press.
  — Textbook treatment of optimization for deep models.

---



