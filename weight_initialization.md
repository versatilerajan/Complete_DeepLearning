# Weight Initialization: Why Starting Values Matter

*Companion notes on why the initial values of a network's weights can make or break training entirely — covering the classic pitfalls (zero, constant, too-small, too-large random initialization) and the two standard fixes (Xavier/Glorot and He), each backed by real trained-network experiments.*

---

## Table of Contents

1. [Why Initialization Matters](#1-why-initialization-matters)
2. [Pitfall 1: Zero Initialization](#2-pitfall-1-zero-initialization)
3. [Pitfall 2: Non-Zero Constant Initialization](#3-pitfall-2-non-zero-constant-initialization)
4. [Real Experiment: Seeing the Symmetry Problem Directly](#4-real-experiment-seeing-the-symmetry-problem-directly)
5. [Pitfall 3: Random Initialization, Too Small](#5-pitfall-3-random-initialization-too-small)
6. [Pitfall 4: Random Initialization, Too Large](#6-pitfall-4-random-initialization-too-large)
7. [Real Experiment: Activation Spread Across Depth](#7-real-experiment-activation-spread-across-depth)
8. [Xavier (Glorot) Initialization](#8-xavier-glorot-initialization)
9. [He Initialization](#9-he-initialization)
10. [Real Experiment: Why ReLU Needs Its Own Formula](#10-real-experiment-why-relu-needs-its-own-formula)
11. [Real Experiment: A Full Training Run, Three Ways](#11-real-experiment-a-full-training-run-three-ways)
12. [Practical Implementation in Keras](#12-practical-implementation-in-keras)
13. [Code: All Initialization Schemes From Scratch](#13-code-all-initialization-schemes-from-scratch)
14. [Key Takeaways](#14-key-takeaways)
15. [Further Reading](#15-further-reading)

---

## 1. Why Initialization Matters

Before the very first training step, every weight in a network needs *some* starting value. This choice sounds trivial but has an outsized effect: poor initialization can cause the exact vanishing/exploding gradient problems covered in the [vanishing gradient notes](vanishing-gradient.md) and [ReLU variants notes](relu-variants.md) before training has even had a chance to get going — or, in the worst cases, prevent the network from ever learning anything at all, no matter how long it trains.

---

## 2. Pitfall 1: Zero Initialization

Setting every weight to `0` seems like a neutral, harmless starting point. It's actually one of the worst possible choices. Here's why: every neuron in a hidden layer receiving the same inputs, with the same (zero) weights, computes the exact same output. During backpropagation (see the [backprop notes](backprop-part1.md)), every one of those neurons then receives the **exact same gradient** — so after any weight update, all of them still have identical weights. This repeats forever: no matter how many neurons a layer has, they all stay perfectly identical throughout the entire training run, which means the layer behaves as if it had just **one single neuron**, no matter how wide it actually is.

---

## 3. Pitfall 2: Non-Zero Constant Initialization

Using some other constant instead of zero (e.g., every weight set to `0.5`) doesn't fix this at all — it has the exact same problem. What matters isn't the specific value, but that **every neuron starts identical**: identical weights still produce identical outputs and identical gradients, so the neurons remain locked together throughout training, just around a different (nonzero) constant instead of zero.

---

## 4. Real Experiment: Seeing the Symmetry Problem Directly

Rather than taking this on faith, here's an actual small network (4 hidden neurons) trained three ways — all-zero weights, all-constant (0.5) weights, and properly randomized weights — with the resulting hidden-layer weight matrix examined directly after training:

![Real experiment: do the hidden neurons learn different things?](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/symmetry_problem.png)

*With zero initialization, every weight stays exactly zero forever — the network never escapes a coin-flip-level loss of 0.693. With constant initialization, all 4 neurons converge to the **exact same** learned weights (visibly identical columns) — the network does learn something, but only as much as a single neuron could, reaching a mediocre loss of 0.313. Only with random initialization do the 4 neurons end up with genuinely different weights (visibly distinct columns), letting the network reach a far better loss of 0.063.*

---

## 5. Pitfall 3: Random Initialization, Too Small

Randomizing weights breaks the symmetry problem — but the *scale* of that randomness matters enormously. If weights are drawn from a distribution with a very small standard deviation (e.g. `0.01`), every layer's output ends up with a much smaller spread than its input. Stack enough layers and this compounds: the signal shrinks geometrically with depth, and with Sigmoid or Tanh activations in particular, activations collapse toward zero — precisely the vanishing gradient mechanism covered in the [vanishing gradient notes](vanishing-gradient.md). With ReLU, small init doesn't cause outright vanishing to zero as severely, but produces an extremely weak, slow-moving training signal.

---

## 6. Pitfall 4: Random Initialization, Too Large

Going the other direction — a large standard deviation (e.g. `1.0` or more) — causes the opposite failure. With Sigmoid or Tanh, large pre-activation values push outputs into the flat, saturated tails of the curve (see the [activation functions notes](activation-functions.md#3-the-classical-three-linear-sigmoid-tanh)), where the derivative is close to zero — again vanishing gradients, just via a different route (saturation instead of shrinkage). With ReLU, large initial weights instead cause the forward signal (and correspondingly the backward gradient) to **grow** with depth, risking numerical instability or outright divergence — the exploding gradient problem.

---

## 7. Real Experiment: Activation Spread Across Depth

To see both failure modes directly, here's a genuine forward pass of random input through a 15-layer Tanh network, tracking the standard deviation of the activations at each layer, for three different initialization scales:

![Real experiment: activation spread through a 15-layer Tanh network](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/activation_stats_tanh.png)

*With small initialization (std=0.01, red), activation spread collapses to essentially zero by the 4th layer — a real, measured vanishing signal. With large initialization (std=1.0, orange), activations saturate near their maximum spread (close to ±1) from the very first layer onward and stay there — the saturation failure mode. Xavier initialization (green), covered next, decays far more gently than either extreme, though even it isn't perfectly flat across 15 layers — matching the original Glorot paper's own claim of approximately (not perfectly) preserving variance.*

---

## 8. Xavier (Glorot) Initialization

Xavier initialization (named for its author, Xavier Glorot) sets the variance of each layer's weights based on that layer's size, specifically chosen so that the variance of the signal is approximately preserved going forward through the network (and going backward through gradients) — directly targeting the small-init/large-init problem from Sections 5–6:

![Xavier and He formulas](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/xavier_he_formulas.png)

```
Normal:   w ~ N(0, 2/(n_in + n_out))
Uniform:  w ~ U(−√(6/(n_in+n_out)), +√(6/(n_in+n_out)))
```

where `n_in` and `n_out` are the number of inputs and outputs of the layer. This was specifically derived assuming a symmetric activation function like **Sigmoid or Tanh**.

---

## 9. He Initialization

**He initialization** (named for Kaiming He) adapts the same variance-preservation idea specifically for **ReLU**. Since ReLU zeroes out roughly half of its inputs (everything negative), the surviving half needs twice the variance to carry the same total signal forward — so He initialization uses exactly double Xavier's variance:

```
Normal:   w ~ N(0, 2/n_in)
Uniform:  w ~ U(−√(6/n_in), +√(6/n_in))
```

---

## 10. Real Experiment: Why ReLU Needs Its Own Formula

Does the distinction actually matter, or is Xavier "close enough" for ReLU too? Testing directly — the same 15-layer forward-pass experiment from Section 7, but now with ReLU activations, comparing Xavier-style scaling against He:

![Real experiment: why ReLU needs He, not Xavier](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/activation_stats_relu.png)

*Using Xavier's formula on a ReLU network (red) still causes a steady, real decay in activation spread — down to less than 1% of its original scale by layer 15. He initialization (green), with its extra factor of 2 to compensate for ReLU's zeroing of negative inputs, keeps the activation spread roughly stable across the entire depth of the network. This is a direct, measured confirmation that the choice of initialization formula genuinely needs to match the activation function being used.*

---

## 11. Real Experiment: A Full Training Run, Three Ways

Finally, putting it all together: an actual 6-hidden-layer ReLU network, trained on a real classification task, with small, large, and He initialization:

![Real experiment: training a 6-hidden-layer ReLU network three ways](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/deep_training_comparison.png)

*Small initialization (red) never escapes random-guessing loss (ln(2) ≈ 0.693) for the entire 1500-epoch run — the signal is too weak from the very first layer to produce any useful gradient. Large initialization (orange) is even worse: the loss shoots past 15 and becomes numerically undefined (NaN) within the first 5 epochs — training doesn't just stall, it collapses outright. He initialization (green) is the only one of the three that actually trains, descending steadily to a loss of 0.033.*

---

## 12. Practical Implementation in Keras

Every Keras layer accepts a `kernel_initializer` argument:

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, activation="tanh", kernel_initializer="glorot_normal", input_shape=(20,)),
    layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
    layers.Dense(64, activation="relu", kernel_initializer="he_uniform"),
    layers.Dense(1, activation="sigmoid", kernel_initializer="glorot_uniform"),
])
```

`glorot_normal`/`glorot_uniform` are Keras's names for Xavier; `he_normal`/`he_uniform` are the He variants. As a rule of thumb: match `glorot_*` with Sigmoid/Tanh layers, and `he_*` with ReLU-family layers (see the [ReLU variants notes](relu-variants.md) for the full family). Notably, `glorot_uniform` is Keras's **default** initializer for `Dense` layers if none is specified — a reasonable general default, though explicitly setting `he_normal` for ReLU layers is better practice.

---

## 13. Code: All Initialization Schemes From Scratch

```python
import numpy as np

def zero_init(n_in, n_out):
    return np.zeros((n_in, n_out))

def constant_init(n_in, n_out, value=0.5):
    return np.full((n_in, n_out), value)

def random_init(n_in, n_out, std, rng):
    return rng.normal(0, std, (n_in, n_out))

def xavier_normal(n_in, n_out, rng):
    std = np.sqrt(2.0 / (n_in + n_out))
    return rng.normal(0, std, (n_in, n_out))

def xavier_uniform(n_in, n_out, rng):
    limit = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-limit, limit, (n_in, n_out))

def he_normal(n_in, n_out, rng):
    std = np.sqrt(2.0 / n_in)
    return rng.normal(0, std, (n_in, n_out))

def he_uniform(n_in, n_out, rng):
    limit = np.sqrt(6.0 / n_in)
    return rng.uniform(-limit, limit, (n_in, n_out))

# ---------------------------------------------------------
# Reproduce the activation-spread experiment from Section 7/10
# ---------------------------------------------------------
def forward_pass_stds(activation_fn, init_fn, n_layers=15, width=100, n_samples=200, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, (n_samples, width))
    stds = [a.std()]
    for _ in range(n_layers):
        W = init_fn(width, width, rng)
        a = activation_fn(a @ W)
        stds.append(a.std())
    return stds

relu = lambda z: np.maximum(0, z)
stds_he = forward_pass_stds(relu, he_normal)
stds_naive = forward_pass_stds(relu, lambda ni, no, rng: xavier_normal(ni, no, rng))
print("He init activation std by layer:   ", [f"{s:.3f}" for s in stds_he])
print("Xavier-on-ReLU std by layer:        ", [f"{s:.3f}" for s in stds_naive])
```

---

## 14. Key Takeaways

- **Zero and constant initialization** both cause a symmetry problem: every neuron in a layer starts (and stays) identical, collapsing an N-neuron layer down to the learning capacity of a single neuron — confirmed directly in a real experiment where 4 neurons converged to visibly identical weights.
- **Random initialization** breaks that symmetry, but the *scale* matters: too small causes activations (and gradients) to vanish with depth; too large causes saturation (Sigmoid/Tanh) or explosion (ReLU) — both confirmed with real forward-pass measurements across a 15-layer network.
- **Xavier/Glorot initialization** (`variance = 2/(n_in+n_out)`) is designed for symmetric activations like Sigmoid and Tanh, approximately preserving signal variance through depth.
- **He initialization** (`variance = 2/n_in`, exactly double Xavier's) is designed specifically for ReLU, accounting for the fact that ReLU zeroes out roughly half its inputs — a real experiment confirms Xavier alone still causes decay on a ReLU network, while He keeps activation spread stable.
- A full real training run makes the practical stakes unmistakable: small init never learns at all, large init diverges to NaN within 5 epochs, and matched (He) initialization trains cleanly to a low loss — from the exact same architecture and data.

---

## 15. Further Reading

- Glorot, X. & Bengio, Y. (2010). *Understanding the Difficulty of Training Deep Feedforward Neural Networks* — the original Xavier/Glorot initialization paper.
- He, K. et al. (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification* — the He initialization paper (same paper that introduced PReLU, covered in the [ReLU variants notes](relu-variants.md)).
- Keras documentation on [initializers](https://keras.io/api/layers/initializers/) for the full list of available schemes and their exact formulas.
- See also this repo's [`vanishing-gradient.md`](vanishing-gradient.md) (the failure mode this document's Sections 5–7 directly cause) and [`mlp.md`](mlp.md) (the forward-pass mechanics these experiments build on).

---

*Diagrams in this document were generated programmatically — including four real trained-network / forward-pass experiments measuring the symmetry problem, activation spread across depth, and full training outcomes under different initializations — and are hosted in this repo's [`images/`](https://github.com/versatilerajan/deepcontent/tree/main/images) folder.*
