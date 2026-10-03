# Feature Scaling: Standardization and Normalization

Age ranges from about 18 to 70. Salary ranges from about 20,000 to 220,000. Feed both into a
neural network side by side and the optimizer sees two directions of wildly different
curvature — one feature can move the loss a lot with a tiny weight change, the other barely
moves it at all unless its weight is enormous. This document makes that claim exact rather
than just illustrative: it computes the actual Hessian of a real regression on synthetic
age/salary data, measures its real condition number before and after scaling, runs real
gradient descent and a real two-layer neural network on both versions, and uses actual
`scikit-learn` (installed in this environment, so run directly rather than just described) for
the practical implementation. Two things came out differently than expected and are reported
as measured: a second, distinct reason scaling matters (it keeps weight initialization valid,
independent of the condition-number story), and a leakage experiment whose statistical
shift was real but whose effect on accuracy was not, in this setup.

This directly extends the condition-number framework built in
[momentum-optimization.md](momentum-optimization.md) and [adam.md](adam.md) — everything there
about $\kappa = \lambda_{\max}/\lambda_{\min}$ and the convergence rate $(\kappa-1)/(\kappa+1)$
is reused here, applied to real tabular data instead of a synthetic quadratic. See also
[gradient-descent.md](gradient-descent.md).

---

## Table of Contents

1. [The problem, concretely](#1-the-problem-concretely)
2. [Standardization and normalization](#2-standardization-and-normalization)
3. [The real Hessian of a real regression](#3-the-real-hessian-of-a-real-regression)
4. [What that condition number does to gradient descent](#4-what-that-condition-number-does-to-gradient-descent)
5. [One learning rate, three outcomes](#5-one-learning-rate-three-outcomes)
6. [A real neural network, and a second reason scaling matters](#6-a-real-neural-network-and-a-second-reason-scaling-matters)
7. [The train/test leakage pitfall](#7-the-traintest-leakage-pitfall)
8. [Practical implementation: scikit-learn](#8-practical-implementation-scikit-learn)
9. [Practical guidance](#9-practical-guidance)
10. [Key takeaways](#10-key-takeaways)
11. [Further reading](#11-further-reading)

---

## 1. The problem, concretely

A neural network's first layer computes $z = Wx + b$. If $x$ has two features on very
different scales, the weights multiplying each feature have to live on correspondingly
different scales too, just to produce comparable outputs — a weight on salary that matters
might need to be around $10^{-4}$, while a weight on age that matters might need to be around
$1$. Gradient descent updates every weight with the **same** learning rate, so a single
$\eta$ is almost never right for both at once: large enough to move the salary weight
usefully and it blows up the age weight; small enough to be safe for the age weight and the
salary weight crawls.

That is the informal version. [Sections 3–4](#3-the-real-hessian-of-a-real-regression) make
it exact: this is precisely the ill-conditioning problem from
[momentum-optimization.md § 3](momentum-optimization.md#3-why-plain-gradient-descent-zig-zags),
where the condition number $\kappa$ of the loss's Hessian sets both the largest stable
learning rate and the number of steps needed to converge. Unscaled features create an
extreme $\kappa$ as a direct, computable consequence of how different their variances are —
not a vague "symmetry" argument.

---

## 2. Standardization and normalization

**Standardization** (z-score scaling) centers each feature at zero and rescales it to unit
variance:

$$x' = \frac{x - \mu}{\sigma}$$

using the feature's own mean $\mu$ and standard deviation $\sigma$. The result has mean 0 and
standard deviation 1, with no fixed upper or lower bound — a feature with outliers still has
those outliers, just rescaled.

**Normalization** (min-max scaling) rescales each feature linearly into a fixed range,
usually $[0, 1]$:

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

The result is bounded, but — as [section 3](#3-the-real-hessian-of-a-real-regression) shows —
that doesn't automatically make it better conditioned than standardization; min-max scaling is
sensitive to outliers (a single extreme value stretches $x_{\max}-x_{\min}$ and compresses
everything else), and it does not equalize variance the way standardization does.

---

## 3. The real Hessian of a real regression

Take two synthetic features, `age` (uniform 18–70) and `salary` (correlated with age, roughly
\$20,000–\$220,000, modeling a realistic case where features aren't independent), and a
continuous target that's a genuine linear function of both plus noise — so ordinary linear
regression is exactly the right model, and its Hessian is exactly computable, not
approximated:

$$L(w) = \frac{1}{N}\|Xw - y\|^2, \qquad \nabla^2 L = \frac{2}{N}X^\top X$$

![Raw feature scales](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_raw_feature_scales.png)

*The two real features used for every experiment in this document: salary's spread is roughly
1,300× age's.*

**Experiment (measured).** Computing $X^\top X$ directly from the data and taking its
eigenvalues:

| | eigenvalues | condition number $\kappa$ |
|---|---|---|
| raw (unscaled) | 67.2, 10,337,800,000 | **153,830,000** |
| standardized | 0.152, 3.848 | **25.3** |
| normalized | 0.0135, 1.384 | **102.7** |

![Condition number](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_condition_number.png)

*The condition number of the exact same regression problem, changed by six orders of
magnitude purely by how the two input features are scaled — nothing about the underlying
relationship between age, salary, and the target changed at all. Notice normalization, at
κ=102.7, is worse-conditioned than standardization here, not better — a real, measured
result, not a typo. Min-max scaling only guarantees both features span $[0,1]$; it says
nothing about matching their variances, and age and salary have correlated, differently-shaped
distributions (age uniform, salary roughly normal with clipping) that scale differently
relative to their range than relative to their standard deviation.*

![Loss contours](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_loss_contours.png)

*The actual loss surface for this regression, each panel zoomed to show comparable contour
spacing (the two panels are not at the same physical scale — see the axis numbers). At
κ=153,830,000 the ellipse is so extreme it is visually indistinguishable from parallel
straight lines — that degenerate appearance is not a plotting error, it is what "elongated
non-symmetrical landscape" looks like at this condition number. The standardized version, at
κ=25.3, is recognizably elliptical, which is what the video means by "more symmetrical."*

---

## 4. What that condition number does to gradient descent

Two quantities depend directly on $\kappa$, both established in
[momentum-optimization.md § 3](momentum-optimization.md#3-why-plain-gradient-descent-zig-zags):
the largest learning rate for which gradient descent is even stable,
$\eta_{\max} = 2/\lambda_{\max}$, and the number of steps needed to converge, which scales
with $\kappa$.

**Experiment (measured).** Computing $\eta_{\max}$ directly from each version's actual
largest eigenvalue, then running real gradient descent at 0.5×, 0.95×, and 1.05× of it:

| | $\eta_{\max} = 2/\lambda_{\max}$ |
|---|---|
| raw | **1.93 × 10⁻¹⁰** |
| standardized | **0.520** |
| normalized | **1.445** |

![Max stable learning rate](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_max_stable_learning_rate.png)

*Raw features force a learning rate roughly **2.7 billion times** smaller than standardized
features need, just to avoid diverging. At 1.05× the stability threshold, every version
diverges (the dotted lines climbing away) — that part of gradient descent's theory doesn't
care about scaling. What scaling changes is what number $\eta_{\max}$ actually is, by nine
orders of magnitude in this case. Within the 40 steps shown, raw's 0.5× and 0.95× curves are
visually flat — not because they're stuck, but because at $\kappa=153{,}830{,}000$ meaningful
progress along the flat eigendirection takes vastly more than 40 steps, which the next
measurement makes precise.*

**Experiment (measured).** Steps needed to get within 1% of the true optimum, each version
using its own best-tuned learning rate (0.9× its $\eta_{\max}$):

| | steps to converge |
|---|---|
| raw | **did not converge within 20,000 steps** |
| standardized | **21** |
| normalized | **261** |

![Steps to converge](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_steps_to_converge.png)

*Even given its own best possible learning rate — the single most favorable setting raw
features could have — gradient descent on unscaled age/salary data does not get within 1% of
the right answer in 20,000 steps. Standardized features need 21. Normalized features, despite
being correctly implemented and bounded to $[0,1]$, still need over 12× as many steps as
standardized, directly because of the higher condition number measured in
[section 3](#3-the-real-hessian-of-a-real-regression).*

---

## 5. One learning rate, three outcomes

The experiments above gave every version its own best learning rate. In practice, nobody
tunes a separate learning rate per feature scale before discovering the problem — someone
picks one value and tries it.

**Experiment (measured).** The same $\eta = 10^{-4}$ — a value that looks unremarkable, and is
in fact close to the standardized version's own eventual optimum — applied to all three:

![Same learning rate, three outcomes](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_same_lr_three_ways.png)

*The loss on raw features goes from 4,733 to 3.4 × 10¹⁴ in a single step and reaches `NaN` by
step 53. The identical learning rate leaves standardized and normalized features training
normally. This is the practical failure mode the video describes as "instability": not a
subtle slowdown, but outright numerical divergence, for a learning rate that would have been
completely reasonable had the inputs been scaled.*

---

## 6. A real neural network, and a second reason scaling matters

Everything above is linear regression, where the loss surface is exactly quadratic and the
condition-number story is exact. Does the same problem show up in an actual neural network? A
real two-layer network (`Dense(2→8) → ReLU → Dense(8→1)`, trained with a real, numerically
verified backward pass — gradient-checked against finite differences to 8 significant figures)
was trained on a genuine binary classification task (loan approval, a noisy logistic function
of age and salary) with raw and standardized inputs.

**Experiment (measured).** Standardized features trained at $\eta=0.1$; raw features were
given the *best* of four learning rates tried (down to $3\times10^{-9}$, since anything larger
diverges immediately):

| | learning rate | final training loss | test accuracy |
|---|---|---|---|
| raw (best of 4 lrs tried) | $10^{-7}$ | 0.693 | **62%** |
| standardized | 0.1 | 0.467 | **76%** |

![Real MLP comparison](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_real_mlp_comparison.png)

*0.693 is exactly $\ln 2$ — the loss of predicting 50/50 for every example. The raw-feature
network never learns anything at its best available learning rate; the standardized one
converges smoothly to a real, working classifier.*

This result raised an obvious question: is the raw network just stuck in the same
slow-convergence regime as the linear case, needing an even smaller learning rate, or is
something else going on? A wider sweep answered it.

**Experiment (measured).** 25 learning rates spanning four orders of magnitude, all tried on
the raw-feature network:

![Learning rate sensitivity](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/08_raw_lr_sensitivity.png)

*Not one of the 25 learning rates reached the standardized network's loss of 0.467. Every
single one lands on one of exactly two values: 17.08 (confidently wrong on nearly every
example) for the six smallest rates, or 0.693 (exactly random-guessing) for the other
nineteen. There is no learning rate in this range that makes the raw-feature network learn.*

**Experiment (measured), diagnosis.** Inspecting the very first forward pass — before any
training step — of each network:

| | first-layer pre-activation range | fraction of ReLU units alive |
|---|---|---|
| raw | $-230{,}000$ to $+4{,}000$ | **12.5%** |
| standardized | $-4$ to $+3$ | **67.5%** |

This is a **second, distinct** reason scaling matters, separate from the condition-number
argument in [sections 3–5](#3-the-real-hessian-of-a-real-regression): standard weight
initialization schemes (He, Xavier) are derived assuming roughly unit-scale inputs, so that
the variance of each layer's output stays controlled going forward and backward. Feed them a
feature with a raw standard deviation of nearly 20,000 and the very first forward pass already
saturates — here, pushing 87.5% of the first layer's ReLU units permanently negative before a
single gradient step has been taken. No learning rate can fix a unit that starts, and stays,
dead. Unlike the condition-number problem, this one is not about *how fast* training
converges; it is about whether the network can get off the ground at all.

---

## 7. The train/test leakage pitfall

A scaler's mean and standard deviation (or min/max) must be computed from the **training**
data only, then applied unchanged to validation and test data — fitting on the full dataset
leaks test-set information into training.

**Experiment (measured).** Fitting `StandardScaler` on the training split only (correct)
versus on the full train+test data (leakage):

| | age mean | salary mean |
|---|---|---|
| fit on train only | 43.51 | 67,206.58 |
| fit on train + test (leaked) | 43.30 | 66,453.81 |
| % difference | 0.48% | 1.12% |

![Leakage pitfall](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/09_leakage_pitfall.png)

*The leaked statistics are measurably different from the correct ones — this is a real
methodological error, not a theoretical nitpick. But its effect on the downstream model in
this experiment was negligible: a model trained with correctly-fit scaling and one trained with
leaked scaling both reached **76%** test accuracy, identically. The reason is specific to this
setup, not a general excuse to skip the correct procedure: with 400 i.i.d. samples split
300/100 from the same generating process, the train and test subsets have very similar
statistics to begin with, so leaking 25% more data into the mean/std estimate barely moves
them. The risk is much larger exactly when train and test differ more — a small dataset, a
test set from a different time period or population, or any real-world distribution shift —
which is precisely when getting this step right matters most and is hardest to verify by
just checking whether accuracy moved.*

---

## 8. Practical implementation: scikit-learn

`scikit-learn` is installed in this environment, so every call below was actually run against
the real training data used throughout this document — these are not illustrative numbers.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Fit on TRAINING data only (section 7)
scaler = StandardScaler()
scaler.fit(X_train)                  # learns mean_ and scale_ (std) from X_train

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)     # SAME mean_/scale_ applied, not refit
```

**Verified output**, printed directly from the fitted scaler and the transformed data:

```
scaler.mean_             -> [43.51, 67206.58]      # [age, salary]
scaler.scale_            -> [15.06, 20208.05]       # standard deviations
X_train_scaled.mean(0)   -> [0.0, -0.0]             # exactly 0, as the definition requires
X_train_scaled.std(0)    -> [1.0, 1.0]              # exactly 1
```

The equivalent for min-max normalization:

```python
scaler = MinMaxScaler()              # default range is [0, 1]
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Verified output:**

```
scaler.data_min_          -> [18.30, 20887.77]
scaler.data_max_          -> [69.95, 120897.67]
X_train_scaled.min(0)     -> [0.0, 0.0]
X_train_scaled.max(0)     -> [1.0, 1.0]
```

Note `X_test_scaled` is **not** guaranteed to land inside $[0,1]$ exactly — if the test set
contains a value outside the training range (an age or salary the model never saw), it maps
to something below 0 or above 1, which is expected and correct, not a bug.

The common pattern inside a full pipeline, fitting only on the training fold:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

scaler = StandardScaler().fit(X_train)          # fit on TRAIN ONLY
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)                # test set never seen during fit

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

---

## 9. Practical guidance

- **Scale every numeric input feature before training a neural network.** Per
  [section 3](#3-the-real-hessian-of-a-real-regression), this is not a minor speed
  optimization — unscaled features changed this document's condition number by six orders of
  magnitude.
- **Default to standardization over normalization** unless a bounded range is specifically
  required (e.g. pixel values feeding a sigmoid, or an algorithm that explicitly assumes
  $[0,1]$ inputs). [Section 3](#3-the-real-hessian-of-a-real-regression) measured
  standardization giving a *better* condition number than normalization on correlated
  features — not guaranteed in general, but a reason not to assume min-max scaling is the
  safer default.
- **Always fit the scaler on training data only**, then apply it unchanged to validation and
  test data ([sections 7–8](#7-the-traintest-leakage-pitfall)). Do this even when, as in
  section 7's experiment, the accuracy impact turns out to be small — the risk is largest
  exactly in the cases (small or non-i.i.d. data) where you can't rely on that being true.
- **Don't assume "it's not converging" is purely a learning-rate problem.**
  [Section 6](#6-a-real-neural-network-and-a-second-reason-scaling-matters) found a case where
  no learning rate fixed a raw-feature network, because the real problem was dead ReLU units
  from the very first forward pass — scaling, not a different $\eta$, was the fix.
- **Scale continuous features; think separately about categorical ones.** One-hot encoded
  categorical features are already on a $\{0,1\}$ scale and don't need standardizing; binary
  indicator features generally shouldn't be standardized either, since doing so turns an
  interpretable 0/1 flag into an arbitrary-looking pair of values without changing what the
  network can represent.
- **Store the fitted scaler with the model**, not just its parameters — at inference time, new
  raw inputs must go through the exact same `transform` the model was trained on.

---

## 10. Key takeaways

1. Unscaled features make the Hessian of a real regression problem on this document's
   synthetic age/salary data have a condition number of **153,830,000**, against **25.3** for
   standardized and **102.7** for normalized features — computed directly from $X^\top X$, not
   estimated.
2. That condition number sets the largest stable learning rate exactly: **1.93 × 10⁻¹⁰** for
   raw features versus **0.520** for standardized — roughly 2.7 billion times smaller.
3. It also sets convergence speed: even at its own best learning rate, raw-feature gradient
   descent did not reach 1% of the true optimum within 20,000 steps; standardized took 21.
4. A single fixed learning rate that is perfectly reasonable for scaled data (here, $10^{-4}$)
   made raw-feature gradient descent diverge to `NaN` within 53 steps.
5. A real two-layer neural network, gradient-checked and genuinely trained, confirmed the same
   pattern: 62% test accuracy for the best of four tried learning rates on raw features versus
   76% for standardized, at a cost of a learning rate roughly a million times smaller.
6. A systematic sweep of 25 learning rates found that raw-feature training *never* matched the
   standardized network's result — the diagnosis was not slow convergence but **87.5% of
   first-layer ReLU units dead from initialization**, because weight-initialization schemes
   assume roughly unit-scale inputs. This is a second, independent reason scaling matters,
   distinct from the condition-number argument.
7. Fitting a scaler on train+test combined (leakage) measurably shifted its statistics (0.48%
   and 1.12% mean differences) but did not change downstream test accuracy in this experiment
   (76% either way) — because train and test were drawn i.i.d. from the same distribution; the
   correct train-only procedure should still always be followed, since that similarity can't
   be assumed in general.

---

## 11. Further reading

- LeCun, Y., Bottou, L., Orr, G. B., & Müller, K.-R. (1998, revised 2012). *Efficient
  BackProp.* In Neural Networks: Tricks of the Trade. — The classical reference for input
  normalization's effect on neural network training dynamics.
- Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by
  Reducing Internal Covariate Shift.* ICML. — Extends the scaling argument from inputs to
  every layer's activations; see also [lenet5.md](lenet5.md) and
  [pretrained-models.md](pretrained-models.md).
- Glorot, X., & Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward
  neural networks.* AISTATS. — The weight-initialization theory behind
  [section 6](#6-a-real-neural-network-and-a-second-reason-scaling-matters)'s dead-ReLU
  diagnosis.
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Delving Deep into Rectifiers.* ICCV. — The
  specific initialization scheme (He initialization) used throughout this notes series,
  including in the network tested in
  [section 6](#6-a-real-neural-network-and-a-second-reason-scaling-matters), derived under the
  same unit-scale-input assumption this document tests.
- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*, chapter 9. — The general theory
  of condition number and gradient descent convergence rate applied throughout
  [sections 3–4](#3-the-real-hessian-of-a-real-regression).
- scikit-learn documentation: `sklearn.preprocessing.StandardScaler` and
  `sklearn.preprocessing.MinMaxScaler` — the exact APIs run in
  [section 8](#8-practical-implementation-scikit-learn).
- Kuhn, M., & Johnson, K. (2019). *Feature Engineering and Selection*, chapter 6. — A broader,
  applied treatment of when and how to scale features, including categorical variables.

---

*Part of a series of deep-learning study notes. Related:
[gradient-descent.md](gradient-descent.md) ·
[momentum-optimization.md](momentum-optimization.md) ·
[adam.md](adam.md) ·
[lenet5.md](lenet5.md) ·
[pretrained-models.md](pretrained-models.md)*
