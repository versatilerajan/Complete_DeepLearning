# Exponentially Weighted Moving Average (EWMA)

*Companion notes on the mathematical building block behind Momentum, RMSprop, and Adam — what EWMA is, how the β parameter trades off noise against lag, the cold-start bias problem and its fix, all backed by real computed experiments on noisy trend data.*

---

## Table of Contents

1. [What Is EWMA?](#1-what-is-ewma)
2. [The Formula](#2-the-formula)
3. [Why "Exponentially Weighted"](#3-why-exponentially-weighted)
4. [Real Experiment: The Effect of β](#4-real-experiment-the-effect-of-β)
5. [The "≈1/(1−β) Days" Intuition](#5-the-11β-days-intuition)
6. [The Cold-Start Problem and Bias Correction](#6-the-cold-start-problem-and-bias-correction)
7. [Real Experiment: Bias Correction in Action](#7-real-experiment-bias-correction-in-action)
8. [Why This Matters for Deep Learning](#8-why-this-matters-for-deep-learning)
9. [Practical Implementation With Pandas](#9-practical-implementation-with-pandas)
10. [Code: EWMA From Scratch](#10-code-ewma-from-scratch)
11. [Key Takeaways](#11-key-takeaways)
12. [Further Reading](#12-further-reading)

---

## 1. What Is EWMA?

An **Exponentially Weighted Moving Average** is a technique for finding the underlying trend in noisy time-series data — daily temperatures, stock prices, or (as covered in [Section 8](#8-why-this-matters-for-deep-learning)) the sequence of gradients seen during neural network training. The core idea: today's smoothed value is a blend of today's raw observation and yesterday's smoothed value, with newer data always counting for more than older data, and the influence of any given data point fading out gradually rather than dropping off a cliff.

---

## 2. The Formula

```
V_t = β·V_{t-1} + (1−β)·θ_t
```

where `θ_t` is today's raw data point, `V_t` is today's smoothed value, `V_{t-1}` is yesterday's smoothed value, and `β ∈ [0, 1)` is a constant controlling how much weight goes to the past versus the present.

![The EWMA formula and how weights decay](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/ewma_formula_and_weights.png)

---

## 3. Why "Exponentially Weighted"

Unrolling the recursion from Section 2 reveals what's really happening:

```
V_t = (1−β)·[θ_t + β·θ_{t-1} + β²·θ_{t-2} + β³·θ_{t-3} + ...]
```

Every past data point contributes something — but each one further back gets an extra factor of `β` multiplied in, so the weights shrink **geometrically** (exponentially) with age, as shown in the right panel of the figure above. A larger `β` (closer to 1) makes that decay much slower, so more history effectively survives into today's average; a smaller `β` makes old data vanish almost immediately.

---

## 4. Real Experiment: The Effect of β

To see this trade-off directly, here's synthetic "temperature" data (a real seasonal trend plus random daily noise), smoothed with four different values of `β`:

![Real experiment: how β trades off noise vs. lag](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/beta_effect_real.png)

*A small `β` (0.1) barely smooths anything — the curve still tracks the noise closely. A moderate `β` (0.5) gives the lowest error against the true trend in this experiment, smoothing out noise while still tracking real changes. Large `β` values (0.9, 0.98) produce a very smooth curve, but visibly **lag behind** the true trend's peaks and troughs — by β=0.98, the curve is so heavily averaged over the past ~50 days that it barely reacts to the actual seasonal swings at all. There is no universally "best" β — it depends entirely on how fast the true underlying signal changes versus how noisy the measurements are.*

---

## 5. The "≈1/(1−β) Days" Intuition

A useful rule of thumb: an EWMA with parameter `β` behaves roughly like an ordinary average over the last `1/(1−β)` data points.

- `β = 0.5` → averages roughly the last **2** days
- `β = 0.9` → averages roughly the last **10** days
- `β = 0.98` → averages roughly the last **50** days

This matches the experiment directly — the right panel of Section 2's figure shows each `β`'s weight curve crossing into negligible territory right around that many days back, and Section 4's results (window size printed in each subplot title) confirm the lag each window size produces in practice.

---

## 6. The Cold-Start Problem and Bias Correction

Look closely at Section 4's plots for `β = 0.9` and `β = 0.98`: the curve starts at exactly **0**, far below any real data value, before slowly climbing toward the true range. This happens because the recursion needs a starting point, and the conventional choice is `V_0 = 0` — but real data isn't anywhere near zero, so the first several smoothed values are artificially dragged down. This is the **cold-start problem**.

The standard fix is **bias correction**: divide the raw EWMA by `(1 − β^t)`, where `t` is the current time step:

```
V_t_corrected = V_t / (1 − β^t)
```

At `t = 1`, this divides by `(1 − β)` — a large correction that compensates for the very-biased first value. As `t` grows, `β^t → 0`, so the correction factor approaches 1 and fades away automatically once the cold-start effect is no longer relevant.

---

## 7. Real Experiment: Bias Correction in Action

![Real experiment: the cold-start problem and its correction](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bias_correction_real.png)

*With `β = 0.9`, the uncorrected EWMA (red) starts at exactly 0 and takes about 15–20 days to climb into the neighborhood of the true trend (black dotted line) — a real, measurable distortion during exactly the period when the model has the least data to work with. The bias-corrected version (green) starts much closer to the true value from day one, and the two versions converge together as `t` grows and the correction factor fades toward 1.*

---

## 8. Why This Matters for Deep Learning

![Why this document comes before Momentum, RMSprop, and Adam](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/connection_to_optimizers.png)

Every advanced optimizer previewed in the [optimizers introduction](optimizers-part1.md) is, at its core, an application of exactly this formula to a different quantity computed during backpropagation:

- **Momentum** applies an EWMA directly to the **gradient** itself, smoothing out the zigzagging that plain gradient descent suffers in narrow ravines (see the real ravine experiment in the [optimizers notes](optimizers-part1.md#5-real-experiment-the-ravine-problem)).
- **RMSprop** applies an EWMA to the **squared gradient**, tracking how large a given parameter's gradients have recently been, so it can scale that parameter's effective learning rate accordingly.
- **Adam** uses *both* EWMAs simultaneously — one on the gradient, one on the squared gradient — each with bias correction applied exactly as in Section 6, since both start from zero and would otherwise be biased during the first several steps of training.

Once the EWMA formula and its `β` trade-off are familiar, the update rules for all three of these optimizers become far easier to read: they're just this same averaging idea, applied to different signals.

---

## 9. Practical Implementation With Pandas

```python
import pandas as pd
import matplotlib.pyplot as plt

data = pd.Series(noisy_temperatures)  # a real pandas Series of raw readings

ewma_09 = data.ewm(alpha=1 - 0.9, adjust=False).mean()    # beta = 0.9
ewma_09_corrected = data.ewm(alpha=1 - 0.9, adjust=True).mean()  # pandas' default (adjust=True) applies bias correction automatically

plt.plot(data, label="raw data", alpha=0.4)
plt.plot(ewma_09, label="EWMA (uncorrected, beta=0.9)")
plt.plot(ewma_09_corrected, label="EWMA (bias-corrected, beta=0.9)")
plt.legend()
plt.show()
```

Note pandas' `alpha` parameter is `1 − β` in the notation used throughout this document, and its `adjust=True` default (rather than `adjust=False`) already performs the bias correction from Section 6 automatically.

---

## 10. Code: EWMA From Scratch

```python
import numpy as np

def ewma(data, beta):
    """Uncorrected EWMA -- starts at V_0 = 0, has the cold-start bias."""
    v = np.zeros(len(data))
    for i in range(1, len(data)):
        v[i] = beta * v[i-1] + (1 - beta) * data[i]
    return v

def ewma_bias_corrected(data, beta):
    """Bias-corrected EWMA, dividing by (1 - beta^t) at each step."""
    v = ewma(data, beta)
    t = np.arange(1, len(data) + 1)
    return v / (1 - beta**t)

# ---------------------------------------------------------
# Reproduce the Section 4 / 7 experiments
# ---------------------------------------------------------
rng = np.random.default_rng(7)
n_days = 120
t = np.arange(n_days)
true_trend = 20 + 8*np.sin(2*np.pi*t/60) + 0.03*t
noisy = true_trend + rng.normal(0, 3.0, n_days)

for beta in [0.1, 0.5, 0.9, 0.98]:
    v = ewma(noisy, beta)
    mse = np.mean((v[10:] - true_trend[10:])**2)
    print(f"beta={beta}: effective window ~{1/(1-beta):.1f} days, MSE={mse:.2f}")

v_raw = ewma(noisy, beta=0.9)
v_corrected = ewma_bias_corrected(noisy, beta=0.9)
print("\nDay 1  -- raw:", v_raw[1], " corrected:", v_corrected[1], " true:", true_trend[1])
print("Day 10 -- raw:", v_raw[10], " corrected:", v_corrected[10], " true:", true_trend[10])
```

---

## 11. Key Takeaways

- EWMA smooths noisy time-series data with `V_t = β·V_{t-1} + (1−β)·θ_t`, giving newer data more weight and letting older data's influence decay geometrically rather than cutting off sharply.
- `β` controls a direct trade-off: higher `β` means more smoothing but more lag behind real changes in the trend; lower `β` tracks changes faster but stays noisier. A real experiment confirmed `β=0.5` (≈2-day window) outperformed both `β=0.1` and `β=0.9`/`0.98` on this particular noisy-but-fast-changing trend.
- A useful rule of thumb: EWMA with parameter `β` behaves like an ordinary average over roughly the last `1/(1−β)` data points.
- Because the recursion starts at `V_0 = 0`, the first several values are artificially biased toward zero — the **cold-start problem**, directly visible in a real experiment.
- **Bias correction** (`V_t / (1 − β^t)`) fixes this, with the correction fading away automatically as `t` grows — also confirmed directly in a real experiment.
- This single formula, applied to different quantities (the gradient itself, or the squared gradient) and combined with bias correction, is the mathematical core of Momentum, RMSprop, and Adam — the next documents in this series.

---

## 12. Further Reading

- Kingma, D. & Ba, J. (2015). *Adam: A Method for Stochastic Optimization* — Section 3 of the original Adam paper derives the bias-correction terms covered here in detail.
- Pandas documentation on [`ewm`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html) for the exact relationship between `alpha`, `span`, `halflife`, and `com` parameterizations.
- See also this repo's [`optimizers-part1.md`](optimizers-part1.md) (the ravine and saddle-point problems that Momentum and RMSprop, built on this document's formula, are designed to fix).

---

*Diagrams in this document were generated programmatically — including real EWMA computations on synthetic noisy trend data at four different β values, and a real demonstration of the cold-start bias problem — and are hosted in this repo's [`images/`](https://github.com/versatilerajan/deepcontent/tree/main/images) folder.*
