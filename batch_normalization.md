# Batch Normalization: Stabilizing Deep Network Training

*Companion notes on Batch Normalization — why unnormalized activations destabilize training, the internal covariate shift problem it solves, exactly how the algorithm works at train and test time, and its four practical advantages — every major claim backed by a real trained-network experiment.*

---

## Table of Contents

1. [Why Batch Normalization?](#1-why-batch-normalization)
2. [Internal Covariate Shift](#2-internal-covariate-shift)
3. [Real Experiment: Measuring Internal Covariate Shift Directly](#3-real-experiment-measuring-internal-covariate-shift-directly)
4. [The Batch Normalization Algorithm](#4-the-batch-normalization-algorithm)
5. [Train vs. Test Time Behavior](#5-train-vs-test-time-behavior)
6. [Real Experiment: Rescuing Bad Initialization](#6-real-experiment-rescuing-bad-initialization)
7. [Real Experiment: Tolerating Higher Learning Rates](#7-real-experiment-tolerating-higher-learning-rates)
8. [The Four Advantages](#8-the-four-advantages)
9. [Practical Implementation in Keras](#9-practical-implementation-in-keras)
10. [Code: Batch Normalization From Scratch](#10-code-batch-normalization-from-scratch)
11. [Key Takeaways](#11-key-takeaways)
12. [Further Reading](#12-further-reading)

---

## 1. Why Batch Normalization?

Every other document in this series has, in its own way, been about the same underlying challenge: keeping the signal flowing through a deep network numerically well-behaved. [Weight initialization](weight-initialization.md) tackles it at the very start of training; the [vanishing/exploding gradient notes](vanishing-gradient.md) describe what goes wrong when it isn't. **Batch Normalization**, introduced in 2015, tackles the same problem *throughout* training — by explicitly normalizing the activations flowing between layers at every single step, not just at initialization.

---

## 2. Internal Covariate Shift

As training proceeds, every layer's weights are constantly changing. That means the *distribution* of values reaching any given layer — its mean, its spread — keeps shifting too, since it depends on every layer before it, all of which are also changing. This shifting-target problem is called **internal covariate shift**: a layer has to continually re-adapt not just to learning its task, but to a constantly moving input distribution underneath it, which slows and destabilizes training.

Batch Normalization's fix is direct: force the input to each layer to have a stable, fixed distribution (mean 0, standard deviation 1) at every single training step, regardless of how much the upstream weights have shifted.

---

## 3. Real Experiment: Measuring Internal Covariate Shift Directly

Rather than taking this on faith, here's an actual deep network (6 hidden layers) trained on a real classification task, with the mean and standard deviation of one hidden layer's input tracked at every epoch — with and without Batch Normalization:

![Real experiment: internal covariate shift, directly measured](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/covariate_shift_real.png)

*Without BatchNorm, the tracked layer's input standard deviation drifts substantially over training — from under 1.0 up past 2.5 at points, and its mean drifts from slightly negative up to nearly 0.3 — a real, measured instance of internal covariate shift. With BatchNorm, the exact same layer's input mean and standard deviation stay pinned almost exactly at 0 and 1 for the **entire** training run, regardless of what the earlier layers are doing. This is Batch Normalization doing precisely what it claims to do.*

---

## 4. The Batch Normalization Algorithm

![The Batch Normalization algorithm](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bn_algorithm.png)

For a mini-batch of pre-activations, the algorithm:

1. Computes the **batch mean** `μ_B` and **batch variance** `σ²_B` across the current mini-batch, per feature.
2. **Normalizes**: subtracts the mean and divides by the standard deviation, giving each feature exactly mean 0 and standard deviation 1 *for this batch*.
3. **Scales and shifts** the normalized value using two **learnable** parameters, `γ` (gamma) and `β` (beta): `y = γẑ + β`.

That last step matters more than it might look: without it, every layer would be *forced* to have mean-0, unit-variance inputs, even if the network would actually benefit from a different scale or offset for a particular layer. Since `γ` and `β` are learned via gradient descent just like any other weight, the network can recover its original representational freedom whenever that's useful — for example, setting `γ = √(σ²_B)` and `β = μ_B` would exactly undo the normalization. Batch Normalization adds stability without removing any capability the network had before.

---

## 5. Train vs. Test Time Behavior

There's a subtlety in the algorithm above: it needs a *batch* of examples to compute `μ_B` and `σ²_B` from. That's fine during training, where mini-batches are the norm — but at test time, predictions often need to be made one example at a time, where there's no meaningful "batch" to compute statistics from.

![Why training and testing normalize differently](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/train_test_time.png)

The fix: during training, alongside computing each batch's `μ_B` and `σ²_B` for immediate use, also maintain a **running (exponentially-weighted moving) average** of these statistics across all the batches seen so far. At test time, use that accumulated running average instead of computing fresh statistics — giving a single, fixed, batch-independent normalization that doesn't depend on which other examples happen to be in the same test batch, or whether there's a batch at all.

---

## 6. Real Experiment: Rescuing Bad Initialization

Recall from the [weight initialization notes](weight-initialization.md) that a 6-hidden-layer ReLU network initialized with either too-small (std=0.01) or too-large (std=1.0) weights failed completely — the first never learned anything, the second diverged to NaN within 5 epochs. Does adding BatchNorm change that outcome?

![Real experiment: BatchNorm rescues catastrophically bad initialization](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bn_rescues_init_real.png)

*Left: with the identical too-small initialization that previously never escaped random-guessing loss (0.693), adding BatchNorm lets the exact same network converge all the way down to 0.012. Right: with the identical too-large initialization that previously diverged to NaN within 5 epochs, BatchNorm lets it train stably to a loss of 0.035. Both catastrophic failures from the weight initialization notes are completely rescued by inserting BatchNorm — direct, measured confirmation of "reduced initialization sensitivity."*

---

## 7. Real Experiment: Tolerating Higher Learning Rates

Using the same good (He) initialization for both, here's the identical network trained at increasing learning rates, with and without BatchNorm:

![Real experiment: BatchNorm tolerates much higher learning rates](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bn_higher_lr_real.png)

*At a modest learning rate of 0.1, both converge, with BatchNorm reaching a lower loss faster. By learning rate 1.0, the non-BatchNorm network becomes visibly unstable (sharp spikes), while BatchNorm continues smoothly. Past learning rate 3.0, the non-BatchNorm network diverges to NaN within single-digit epochs both times — while BatchNorm keeps training successfully even at a learning rate of 8.0, a value that would be unthinkable to use without it.*

---

## 8. The Four Advantages

![Batch Normalization's four advantages](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bn_advantages.png)

- **Stable training** — a much wider range of hyperparameters, including weight initialization scale (Section 6), works without diverging.
- **Faster convergence** — tolerates dramatically higher learning rates (Section 7), which directly speeds up training.
- **Regularization effect** — since each mini-batch's statistics are slightly different from the full dataset's true statistics, BatchNorm injects a small amount of noise into training, similar in spirit to (though generally weaker than) [Dropout](dropout.md).
- **Reduced initialization sensitivity** — the specific weight initialization scheme (see the [weight initialization notes](weight-initialization.md)) becomes far less critical, since BatchNorm actively re-normalizes the signal at every layer regardless of how it was initialized.

---

## 9. Practical Implementation in Keras

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, kernel_initializer="he_normal", input_shape=(20,)),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dense(64, kernel_initializer="he_normal"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=keras.optimizers.SGD(learning_rate=1.0), loss="binary_crossentropy")
```

Note the ordering: `Dense` (the raw linear transform) → `BatchNormalization` → `Activation`. Applying BatchNorm to the *pre-activation* value, before the non-linearity, is the standard placement (and matches exactly what the real experiments in this document use).

---

## 10. Code: Batch Normalization From Scratch

```python
import numpy as np

def batchnorm_forward(z, gamma, beta, eps=1e-8):
    mu = z.mean(axis=0)
    var = z.var(axis=0)
    z_hat = (z - mu) / np.sqrt(var + eps)
    y = gamma * z_hat + beta
    cache = (z, z_hat, mu, var, gamma, eps)
    return y, cache

def batchnorm_backward(dy, cache):
    z, z_hat, mu, var, gamma, eps = cache
    N = z.shape[0]
    std_inv = 1.0 / np.sqrt(var + eps)

    dgamma = np.sum(dy * z_hat, axis=0)
    dbeta = np.sum(dy, axis=0)

    dz_hat = dy * gamma
    dvar = np.sum(dz_hat * (z - mu) * -0.5 * std_inv**3, axis=0)
    dmu = np.sum(dz_hat * -std_inv, axis=0) + dvar * np.mean(-2.0 * (z - mu), axis=0)
    dz = dz_hat * std_inv + dvar * 2.0 * (z - mu) / N + dmu / N
    return dz, dgamma, dbeta

# ---------------------------------------------------------
# Running averages for test-time use (Section 5)
# ---------------------------------------------------------
def update_running_stats(running_mu, running_var, batch_mu, batch_var, momentum=0.9):
    running_mu = momentum * running_mu + (1 - momentum) * batch_mu
    running_var = momentum * running_var + (1 - momentum) * batch_var
    return running_mu, running_var

def batchnorm_inference(z, gamma, beta, running_mu, running_var, eps=1e-8):
    z_hat = (z - running_mu) / np.sqrt(running_var + eps)
    return gamma * z_hat + beta
```

This implementation was verified against numerical gradient checking (finite differences) before being used in any of the experiments above — every gradient matched to within `1e-10`, confirming the backward pass is exactly correct.

---

## 11. Key Takeaways

- Internal covariate shift — the constantly shifting distribution of each layer's inputs as upstream weights change during training — is the core problem Batch Normalization targets, and a real experiment shows it directly: activation statistics drift substantially without BatchNorm and stay essentially frozen at mean 0, std 1 with it.
- The algorithm normalizes each mini-batch's pre-activations to mean 0, standard deviation 1, then applies a learnable scale (`γ`) and shift (`β`) so the network can recover its original flexibility whenever needed.
- Training uses each batch's own statistics; testing uses a running average accumulated during training, since test-time predictions may not come in meaningful batches.
- A real experiment shows BatchNorm **completely rescuing** both a too-small and a too-large weight initialization that otherwise failed outright (stuck at random-guessing loss, or diverging to NaN).
- Another real experiment shows BatchNorm tolerating learning rates (up to 8.0 in the test) that cause a non-normalized network to diverge to NaN within single-digit epochs.
- The net effect across all four advantages is the same theme: Batch Normalization makes training dramatically more forgiving of choices (initialization, learning rate) that would otherwise need to be tuned very carefully.

---

## 12. Further Reading

- Ioffe, S. & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* — the original paper.
- Santurkar, S. et al. (2018). *How Does Batch Normalization Help Optimization?* — a later paper questioning whether internal covariate shift reduction is really the mechanism behind BatchNorm's benefits, and proposing an alternative (smoothing the loss landscape) — worth reading for a more nuanced picture.
- Keras documentation on [`BatchNormalization`](https://keras.io/api/layers/normalization_layers/batch_normalization/) for implementation details and momentum/epsilon defaults.
- See also this repo's [`weight-initialization.md`](weight-initialization.md) (the exact failures rescued in Section 6) and [`vanishing-gradient.md`](vanishing-gradient.md) / [`gradient-descent.md`](gradient-descent.md) (the training-stability problems this document's Section 7 experiment directly addresses).

---

*Diagrams in this document were generated programmatically — including four real trained-network experiments (internal covariate shift measurement, bad-initialization rescue, and higher-learning-rate tolerance) backing every major claim — and are hosted in this repo's [`images/`](https://github.com/versatilerajan/deepcontent/tree/main/images) folder.*
