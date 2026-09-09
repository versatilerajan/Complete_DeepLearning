# Activation Functions: A Complete Tour

*Companion notes covering every commonly-used activation function — what it computes, why it exists, where it wins, where it fails, and how they all stack up against each other — backed by a real trained-network experiment, not just theory.*

---

## Table of Contents

1. [What Are Activation Functions, and Why Do They Exist?](#1-what-are-activation-functions-and-why-do-they-exist)
2. [Properties of an Ideal Activation Function](#2-properties-of-an-ideal-activation-function)
3. [The Classical Three: Linear, Sigmoid, Tanh](#3-the-classical-three-linear-sigmoid-tanh)
4. [The ReLU Family](#4-the-relu-family)
5. [Softmax: The Odd One Out](#5-softmax-the-odd-one-out)
6. [Modern Smooth Activations: Swish and GELU](#6-modern-smooth-activations-swish-and-gelu)
7. [A Real Experiment: Does the Choice Actually Matter?](#7-a-real-experiment-does-the-choice-actually-matter)
8. [Full Comparison Table](#8-full-comparison-table)
9. [Which One Should You Actually Use?](#9-which-one-should-you-actually-use)
10. [Code: Every Activation From Scratch](#10-code-every-activation-from-scratch)
11. [Key Takeaways](#11-key-takeaways)
12. [Further Reading](#12-further-reading)

---

## 1. What Are Activation Functions, and Why Do They Exist?

An activation function is applied to the weighted sum at each neuron (`z = w·x + b`, see the [MLP notes](mlp.md)) to decide how much that neuron actually "fires." Without one, a neural network — no matter how many layers deep — is mathematically equivalent to a **single linear transformation**:

![Why activation functions are necessary](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/why_nonlinearity_necessary.png)

*Left: three stacked layers with no activation function collapse into one straight line — depth adds nothing. Right: inserting a non-linear activation (here, ReLU across a small hidden layer) lets genuinely new, non-linear shapes emerge from the same basic building blocks. This is the entire reason activation functions exist.*

---

## 2. Properties of an Ideal Activation Function

Before covering specific functions, it helps to know what's actually being traded off between them:

![Properties of an ideal activation function](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/ideal_properties.png)

No real function satisfies all five perfectly — every activation function below sacrifices at least one of these in exchange for gains elsewhere, which is why so many different ones exist rather than a single obvious winner.

---

## 3. The Classical Three: Linear, Sigmoid, Tanh

![Linear, Sigmoid, and Tanh: functions and derivatives](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/classical_activations.png)

### Linear (Identity)
```
f(z) = z          f'(z) = 1
```
**Where it wins:** the only sensible choice for a regression output layer, where you actually want an unbounded real number out. Trivially cheap and never saturates.
**Where it fails:** provides zero non-linearity — as shown in Section 1, using it in hidden layers defeats the entire purpose of having multiple layers.

### Sigmoid
```
f(z) = 1/(1+e⁻ᶻ)          f'(z) = f(z)(1−f(z))
```
**Where it wins:** squashes any input into `(0,1)`, making it a natural choice for a **binary classification output layer** where the output should be interpreted as a probability (see the [loss functions notes](loss-functions.md) on Binary Cross-Entropy, which pairs with it).
**Where it fails:** the derivative peaks at only **0.25** and decays to near-zero away from the origin — exactly the mechanism behind the vanishing gradient problem covered in depth in the [vanishing gradient notes](vanishing-gradient.md). It's also not zero-centered (always positive), which biases gradient directions during training and slows convergence.

### Tanh
```
f(z) = tanh(z)          f'(z) = 1 − tanh(z)²
```
**Where it wins:** zero-centered (output ranges over `(−1,1)`, symmetric around 0) — a genuine improvement over sigmoid that leads to better-behaved gradients, which is why it's historically preferred over sigmoid inside hidden layers, especially in RNNs.
**Where it fails:** its derivative still peaks at only 1.0 and still shrinks to near-zero away from the origin — it reduces the vanishing gradient problem compared to sigmoid but doesn't eliminate it.

---

## 4. The ReLU Family

![The ReLU family: ReLU, Leaky ReLU, ELU, SELU](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/relu_family.png)

### ReLU (Rectified Linear Unit)
```
f(z) = max(0, z)          f'(z) = 1 if z>0 else 0
```
**Where it wins:** the industry-standard default for hidden layers (see the [approximation theory](README.md) and [vanishing gradient](vanishing-gradient.md) notes for why). It's essentially free to compute, and its derivative is exactly 1 for any positive input — no shrinking at all in the active region, which is what makes very deep networks trainable in the first place.
**Where it fails:** any neuron that ends up with a negative pre-activation for every training example gets a gradient of exactly 0 forever — it's permanently "dead" and can never recover, since gradient descent has literally nothing to work with. It's also not zero-centered.

### Leaky ReLU
```
f(z) = z if z>0 else αz  (α ≈ 0.01–0.1)
```
**Where it wins:** gives negative inputs a small non-zero slope instead of exactly 0, so a neuron that strays into negative territory still receives *some* gradient and has a chance to recover — directly fixing ReLU's dead-neuron problem.
**Where it fails:** `α` is a fixed hyperparameter chosen by hand — there's no principled way to know the right value in advance, and getting it wrong under- or over-corrects for the dead-neuron problem.

### PReLU (Parametric ReLU)
Same shape as Leaky ReLU, but `α` is **learned** during training rather than fixed. **Where it wins:** removes the guesswork from choosing `α`, adapting per-neuron. **Where it fails:** adds extra trainable parameters, which can encourage overfitting on smaller datasets.

### ELU (Exponential Linear Unit)
```
f(z) = z if z>0 else α(eᶻ−1)
```
**Where it wins:** smoothly saturates to `−α` for negative inputs rather than a hard linear slope, which empirically tends to push mean activations closer to zero (better than ReLU's zero-centering problem) and can lead to faster, more stable convergence.
**Where it fails:** computing `eᶻ` is meaningfully more expensive than ReLU's simple max operation, at the scale of billions of activations per forward pass.

### SELU (Scaled ELU)
```
f(z) = λz if z>0 else λα(eᶻ−1)         (α≈1.6733, λ≈1.0507, specific fixed constants)
```
**Where it wins:** designed so that, under specific initialization and architecture conditions, activations across layers automatically converge toward zero mean and unit variance — "self-normalizing," without needing separate batch normalization.
**Where it fails:** the self-normalizing property only holds under those specific conditions (a particular weight initialization, fully-connected architecture) — it doesn't reliably transfer to arbitrary architectures like CNNs with the same guarantee.

---

## 5. Softmax: The Odd One Out

Every function above operates on one number at a time. **Softmax** is fundamentally different — it takes an entire vector of raw scores (logits) and converts them jointly into a probability distribution:

![Softmax: the one activation that looks at all outputs together](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/softmax_diagram.png)

```
softmax(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)
```

**Where it wins:** it's the standard choice for the output layer of any multi-class classifier — the outputs are guaranteed non-negative and sum to exactly 1, so they can be directly interpreted as class probabilities (see the [loss functions notes](loss-functions.md) on Categorical Cross-Entropy, its standard pairing).
**Where it fails:** it only makes sense as an output-layer function — using it in a hidden layer would be unusual and counterproductive, since it destroys the individual magnitude information of each neuron in favor of relative comparison.

---

## 6. Modern Smooth Activations: Swish and GELU

![Swish and GELU compared to ReLU](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/modern_smooth_activations.png)

### Swish / SiLU
```
f(z) = z · σ(z)
```
**Where it wins:** smooth and non-monotonic (it dips slightly negative before rising, unlike ReLU's hard corner at 0), which empirically improves optimization in some deep networks (it was discovered via automated search and used in the EfficientNet architecture).
**Where it fails:** meaningfully more expensive than ReLU (requires computing a sigmoid), and the improvement over ReLU is often small enough that it's not always worth the extra compute.

### GELU (Gaussian Error Linear Unit)
```
f(z) = z · Φ(z)          (Φ = standard normal CDF)
```
**Where it wins:** the default choice in most modern Transformer architectures (BERT, GPT, and most successors) — its smooth, probabilistic weighting of each input tends to work very well at the scale these models operate at.
**Where it fails:** the most computationally expensive function in this entire document (involves the error function or a tanh-based approximation), which matters when you're evaluating it billions of times.

---

## 7. A Real Experiment: Does the Choice Actually Matter?

Rather than taking the vanishing-gradient story on faith, here's an actual small network — same architecture, same data, same random seed, same everything **except the hidden-layer activation function** — trained on a real non-linear classification task (the classic two-moons dataset):

![Real experiment: same network, different activations](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/real_activation_comparison.png)

*Tanh, ReLU, and Leaky ReLU all converge to a low loss within a few hundred epochs. Sigmoid, however, visibly plateaus around loss ≈ 0.31–0.35 for roughly the first 800 epochs before finally starting to descend — a direct, measured consequence of its small, saturating derivative slowing gradient flow through the network. This isn't a theoretical curiosity; it's the actual, real difference in training dynamics that theory predicts.*

---

## 8. Full Comparison Table

![Full activation function comparison table](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/comparison_table.png)

---

## 9. Which One Should You Actually Use?

- **Hidden layers, general default:** ReLU. It's cheap, doesn't saturate for positive inputs, and works well in the overwhelming majority of architectures.
- **ReLU giving you dead neurons?** Try Leaky ReLU or PReLU first — minimal extra cost, direct fix for that specific problem.
- **Want potentially faster convergence and can afford the compute?** ELU or Swish are reasonable upgrades over plain ReLU.
- **Building or fine-tuning a Transformer?** GELU is the de facto standard — most pretrained models you'd build on already use it.
- **RNN hidden state?** Tanh remains the traditional choice, for its zero-centered output.
- **Binary classification output layer:** Sigmoid.
- **Multi-class classification output layer:** Softmax.
- **Regression output layer:** Linear (no activation at all).

---

## 10. Code: Every Activation From Scratch

```python
import numpy as np
from scipy.special import erf

# ---------------------------------------------------------
# Classical
# ---------------------------------------------------------
def linear(z): return z
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def tanh(z): return np.tanh(z)

# ---------------------------------------------------------
# ReLU family
# ---------------------------------------------------------
def relu(z): return np.maximum(0, z)
def leaky_relu(z, alpha=0.01): return np.where(z > 0, z, alpha * z)
def elu(z, alpha=1.0): return np.where(z > 0, z, alpha * (np.exp(z) - 1))
def selu(z, alpha=1.6733, lam=1.0507): return np.where(z > 0, lam*z, lam*alpha*(np.exp(z)-1))

# ---------------------------------------------------------
# Output-layer / modern
# ---------------------------------------------------------
def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)   # numerical stability
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)

def swish(z): return z * sigmoid(z)
def gelu(z): return 0.5 * z * (1 + erf(z / np.sqrt(2)))

# ---------------------------------------------------------
# Corresponding derivatives (needed for backprop)
# ---------------------------------------------------------
def sigmoid_deriv(a): return a * (1 - a)              # a = sigmoid(z), already computed
def tanh_deriv(a): return 1 - a**2                     # a = tanh(z)
def relu_deriv(z): return (z > 0).astype(float)
def leaky_relu_deriv(z, alpha=0.01): return np.where(z > 0, 1.0, alpha)
def elu_deriv(z, a, alpha=1.0): return np.where(z > 0, 1.0, a + alpha)  # a = elu(z)
```

**Using Keras**, every activation is just a string (or layer) argument:

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(20,)),
    layers.Dense(64, activation="gelu"),        # or "swish", "elu", "selu", "tanh" ...
    layers.Dense(3, activation="softmax"),      # multi-class output
])
```

---

## 11. Key Takeaways

- Activation functions exist to introduce non-linearity — without them, any depth of network is mathematically just one linear transformation.
- No activation function satisfies every desirable property (non-linear, differentiable, cheap, zero-centered, non-saturating) at once — each one below is a different trade-off.
- **Sigmoid** and **Tanh** both saturate and cause vanishing gradients; Tanh is the better of the two (zero-centered), but both are largely replaced by ReLU-family functions in hidden layers today.
- **ReLU** is the practical default: cheap, doesn't saturate for positive inputs, but can produce permanently "dead" neurons.
- **Leaky ReLU, PReLU, ELU, SELU** are all direct attempts to fix ReLU's dead-neuron and zero-centering weaknesses, each with its own added cost or complexity.
- **Softmax** is uniquely vector-valued, reserved for multi-class output layers.
- **Swish and GELU** are smooth, modern alternatives to ReLU; GELU in particular is the standard in Transformer architectures.
- A real trained-network experiment confirms the theory directly: Sigmoid visibly plateaus for hundreds of epochs before Tanh, ReLU, and Leaky ReLU all converge far faster on the exact same task.

---

## 12. Further Reading

- Nair, V. & Hinton, G. (2010). *Rectified Linear Units Improve Restricted Boltzmann Machines* — one of the key papers establishing ReLU's dominance.
- Clevert, D. et al. (2015). *Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)*.
- Klambauer, G. et al. (2017). *Self-Normalizing Neural Networks* — the SELU paper.
- Ramachandran, P. et al. (2017). *Searching for Activation Functions* — the paper that discovered Swish via automated search.
- Hendrycks, D. & Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs)*.
- See also this repo's [`vanishing-gradient.md`](vanishing-gradient.md) (the mechanism behind Sections 3–4's saturation problems) and [`mlp.md`](mlp.md) (where activation functions plug into the forward pass).

---

*Diagrams in this document were generated programmatically — including a real trained-network experiment comparing four activation functions head-to-head — and are hosted in this repo's [`images/`](https://github.com/versatilerajan/deepcontent/tree/main/images) folder.*
