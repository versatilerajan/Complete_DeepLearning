# LeNet-5

LeNet-5 (LeCun, Bottou, Bengio & Haffner, 1998) is where convolution, pooling, and fully
connected layers first came together into the architecture pattern every CNN since has been
a variation on. It is small enough — about 60,000 parameters — to build entirely from scratch,
train end to end, and inspect at every layer, which is what this document does. Every figure
below was produced by actually running that from-scratch network: real forward and backward
passes, a full gradient check against numerical differentiation, real training on a real
(if synthetic — see [section 6](#6-training-lenet-5-on-a-synthetic-task)) image classification
task, and a direct, measured answer to the question the video raises about tanh versus ReLU.
Every number in the prose is that code's output.

This assumes [convolution-operation.md](convolution-operation.md) and
[pooling-operation.md](pooling-operation.md) for the two operations LeNet-5 is built from, and
[gradient-descent.md](gradient-descent.md) / [momentum-optimization.md](momentum-optimization.md)
for how it gets trained. It leads into [cnn-architectures.md](cnn-architectures.md), which
covers what changed between LeNet-5 and the deep networks that followed it.

---

## Table of Contents

1. [The three building blocks, recapped](#1-the-three-building-blocks-recapped)
2. [The architecture, laid out end to end](#2-the-architecture-laid-out-end-to-end)
3. [Output shape at every stage: verified](#3-output-shape-at-every-stage-verified)
4. [Parameter count: verified, and one surprise](#4-parameter-count-verified-and-one-surprise)
5. [Why tanh, and what it costs: measured](#5-why-tanh-and-what-it-costs-measured)
6. [Training LeNet-5 on a synthetic task](#6-training-lenet-5-on-a-synthetic-task)
7. [Receptive field: how far each layer can see](#7-receptive-field-how-far-each-layer-can-see)
8. [Implementation from scratch](#8-implementation-from-scratch)
9. [Practical guidance](#9-practical-guidance)
10. [Key takeaways](#10-key-takeaways)
11. [Further reading](#11-further-reading)

---

## 1. The three building blocks, recapped

LeNet-5 is built from exactly three kinds of layer, each covered in detail elsewhere in this
series and used here without re-deriving:

- **Convolutional layers** ([convolution-operation.md](convolution-operation.md)) slide a
  small learned filter over the input and extract local features — edges and simple textures
  in the earliest layer, more complex combinations deeper in.
- **Pooling layers** ([pooling-operation.md](pooling-operation.md)) shrink the spatial size of
  a feature map with a fixed, parameter-free summary, trading spatial precision for a smaller,
  more position-tolerant representation. LeNet-5 uses **average** pooling throughout, not the
  max pooling that became standard later — a choice with a measurable, if small, effect on
  its final feature maps (revisited in [section 6](#6-training-lenet-5-on-a-synthetic-task)).
- **Fully connected layers** flatten whatever spatial structure remains into a 1D vector and
  mix it freely, exactly like the layers in a plain feedforward network — this is where a CNN's
  spatially-organized features get turned into a final classification decision.

LeNet-5's contribution was not inventing any one of these — convolution and gradient-based
learning both predate it — but demonstrating that stacking them, trained end to end by
backpropagation, could solve a real, previously hard problem: reading handwritten digits on
bank checks, at production scale, for Yann LeCun's group at Bell Labs.

---

## 2. The architecture, laid out end to end

![Architecture diagram](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_architecture_diagram.png)

*Seven layers, alternating convolution and pooling, finished by two fully connected layers
and a softmax. Every shape shown was produced by actually building this exact network in code
and running a real tensor through it — none of it is copied from the paper.*

Reading across: a **32×32** single-channel input (already larger than the 28×28 digits it was
built for — LeCun et al. pad each digit so its strokes never touch the image border, which
matters once [section 7](#7-receptive-field-how-far-each-layer-can-see) looks at receptive
fields) goes through:

1. **C1** — 6 filters, 5×5, no padding → 28×28×6, then tanh
2. **S2** — 2×2 average pool → 14×14×6
3. **C3** — 16 filters, 5×5, no padding → 10×10×16, then tanh
4. **S4** — 2×2 average pool → 5×5×16
5. **C5** — 120 filters, 5×5, no padding → 1×1×120, then tanh (this is a convolution, not a
   dense layer, but because the input is now exactly 5×5 — the filter's own size — it behaves
   identically to a dense layer from a flattened 5×5×16 input; most modern reimplementations
   write it as one)
6. **F6** — dense, 120→84, then tanh
7. **Output** — dense, 84→10 (the original paper used Euclidean RBF units against
   handcrafted class templates here; a softmax + cross-entropy head, used throughout this
   document, is the standard modern substitute and is what essentially every reimplementation,
   including the Keras one in the video, actually uses)

---

## 3. Output shape at every stage: verified

Every spatial transition in LeNet-5 follows the two formulas already established in this
series: convolution shrinks by $n-f+1$
([convolution-operation.md § 5](convolution-operation.md#5-output-size-the-n--f--1-formula)),
and a stride-$f$ pool divides by $f$
([pooling-operation.md § 3](pooling-operation.md#3-memory-measured)).

**Experiment (measured).** Running the actual network at 9 different input sizes from 32 to
64 and comparing the resulting shape at every stage against the two formulas: **all 9 matched
exactly.**

![Shape formula verification](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_shape_formula_verification.png)

*Measured spatial size through the network for nine different starting sizes. Only $n_0=32$
lands exactly on 1×1 by C5 — this is not a coincidence but a design constraint: LeCun et al.
chose 32 specifically so that C5, structurally a convolution, would collapse to a single
spatial position and act as a dense layer. Feed the network anything else and C5 stays a real
spatial map (e.g. 9×9 at $n_0=64$), which changes its behavior and its parameter count.*

---

## 4. Parameter count: verified, and one surprise

**Experiment (measured).** Counting the actual learnable weights and biases in each layer of
the built network:

| layer | parameters | share of total |
|---|---|---|
| C1 (6 filters, 5×5×1) | 156 | 0.3% |
| S2 (average pool) | **0** | 0% |
| C3 (16 filters, 5×5×6) | 2,416 | 3.9% |
| S4 (average pool) | **0** | 0% |
| C5 (120 filters, 5×5×16) | **48,120** | **78.0%** |
| F6 (dense 120→84) | 10,164 | 16.5% |
| Output (dense 84→10) | 850 | 1.4% |
| **Total** | **61,706** | |

![Parameter count](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_parameter_count.png)

*Every value here matches the standard $(f \times f \times C_{in} + 1) \times C_{out}$
formula exactly — e.g. C1: $(5{\times}5{\times}1+1)\times6=156$. This count uses **full**
connectivity between C3 and its input, the convention essentially every modern
reimplementation (including Keras) uses; the original 1998 paper connected each C3 filter to
only a subset of S2's six maps, via a fixed, hand-designed connection table, for reasons of
1998-era compute cost — a detail almost never reproduced today, and dropping it accounts for
essentially all of the gap between this count and the commonly cited "roughly 60,000".*

The surprise, visible directly in the bar chart: **C5, a single layer, holds 78% of the
entire network's parameters.** LeNet-5 is described everywhere as a small, elegant
architecture, and in terms of depth and design it is — but almost all of its capacity sits in
one wide, effectively-fully-connected layer at the boundary between "convolutional" and
"fully connected." The pooling layers, by contrast, contribute exactly zero, for the reasons
established in [pooling-operation.md § 4](pooling-operation.md#4-no-learnable-parameters-verified-not-assumed).

---

## 5. Why tanh, and what it costs: measured

The video flags tanh as a specifically 1998 choice, since ReLU (Nair & Hinton, 2010) did not
yet exist. Both claims in the usual explanation — that tanh saturates and that this causes
vanishing gradients — are tested directly here rather than repeated.

**Saturation, measured.** At a pre-activation standard deviation of 2.0 (a realistic scale for
an untrained or lightly-trained layer), sampled over 20,000 values:

| | tanh | ReLU |
|---|---|---|
| mean derivative | **0.361** | **0.498** |
| fraction "stuck" (tanh: \|output\| > 0.99; ReLU: input ≤ 0) | 18.0% | 50.2% |

![tanh saturation](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_tanh_saturation.png)

*(a) Both functions and their derivatives — tanh's derivative is a bump that decays to
**exactly** zero on both sides; ReLU's derivative is a hard step, exactly zero for any
negative input and exactly one for any positive input, never partial. (b) At this scale, more
ReLU units are fully dead (50.2%, since half of a zero-mean distribution is negative) than
tanh units are fully saturated (18.0%) — but every tanh unit contributes *some* gradient
signal, damped continuously, while a dead ReLU unit contributes exactly zero, permanently,
unless its input distribution shifts.*

**Gradient flow through actual depth, measured.** This is the more direct test, and it depends
on something the saturation numbers alone don't capture: how large the weights are. Two
regimes, same network, same random input and target, only the activation and the weight scale
differ:

![Gradient flow](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_gradient_flow.png)

*(a) With well-scaled (He-like) weights — ordinary, correctly-initialized training — tanh and
ReLU behave almost identically: gradient signal shrinks by a similar factor (tanh: 0.033×,
ReLU: 0.020×) going from the output back to C1. Neither has a meaningful edge here.
(b) With deliberately large, poorly-scaled weights — an untuned or misconfigured network — the
picture reverses sharply. ReLU's gradient **explodes** by 10,038× from output to C1; tanh's
explodes too, but only by 57×, **176 times less**. The mechanism is exactly
[section (a)](#5-why-tanh-and-what-it-costs-measured)'s derivative bound: tanh's derivative can
never exceed 1 and is usually much smaller, which puts an implicit ceiling on how much any
single layer's backward pass can amplify a gradient; ReLU's derivative is exactly 1 wherever a
unit is active, with no such ceiling, so a badly-scaled network with ReLU has nothing to stop
runaway growth.*

The honest summary reverses the usual one-line story. Tanh's saturation is a real cost when
weights are well-scaled and gradients need to travel far — this is the textbook
vanishing-gradient argument, and it is why very deep tanh networks were hard to train before
architectural fixes like residual connections. But saturation is also a **bound**, and that
bound protects tanh networks from a failure mode — gradient explosion under poor
initialization — that unbounded ReLU has no defense against at all. Neither activation is
strictly safer; they fail in different, specific circumstances.

---

## 6. Training LeNet-5 on a synthetic task

This environment has no internet access, so the network below was not trained on real MNIST —
it was trained on a small **synthetic** dataset of five stroke patterns (a loop, and vertical,
horizontal, diagonal, and cross strokes) loosely inspired by digit shapes, generated entirely
by code. This is disclosed here rather than presented as a real MNIST result.

![Synthetic dataset](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_synthetic_dataset.png)

*One example per class from the 350-image training set (70 per class), each 32×32 with a
small amount of pixel noise added.*

**Experiment (measured).** The exact from-scratch LeNet-5 (tanh + average pool) trained
against a modernized variant (ReLU + max pool, everything else identical), 30 epochs, 3 seeds:

| | parameters | final test accuracy | epochs to 90% |
|---|---|---|---|
| LeNet-5 original (tanh, avg pool) | 61,281 | **100%** | 1 |
| modernized (ReLU, max pool) | 61,281 | **100%** | 1 |

![Training comparison](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_lenet_training_comparison.png)

*(a) Training loss over 30 epochs, mean ± 1 sd over 3 seeds. (b) Test accuracy — both variants
solve this task almost immediately. This task is easy enough that it does not discriminate
between the two variants; the meaningful, measured difference between tanh/avg-pool and
ReLU/max-pool already showed up in [section 5](#5-why-tanh-and-what-it-costs-measured)'s
gradient-flow experiment and in [pooling-operation.md § 7](pooling-operation.md#7-max-vs-average-vs-global-pooling)'s
measurement of average pooling diluting sparse peaks. What this experiment does confirm is
simpler but not to be taken for granted: the from-scratch implementation — every conv,
pool, dense, and activation layer, and their backward passes — trains correctly end to end.*

That correctness claim is not asserted lightly: before this training run, the full network's
backward pass was checked against numerical (finite-difference) gradients on a real batch with
a real loss, and matched to 6 significant figures (analytic $-0.0951139$ vs. numerical
$-0.0951139$ on a sampled convolution weight).

![Learned C1 filters](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/08_learned_c1_filters.png)

*C1's six filters after training, starting from random initialization. Backpropagation alone
— the same mechanism verified in
[convolution-operation.md § 10](convolution-operation.md#10-filters-are-learned-not-designed)
— shaped these from noise; nothing here was hand-designed.*

---

## 7. Receptive field: how far each layer can see

A unit's **receptive field** is the set of input pixels that can influence it at all. This
grows automatically as depth increases, and it can be measured directly and exactly: pick one
unit, backpropagate a gradient of 1 from just that unit, and see which input pixels end up
with a nonzero gradient.

**Experiment (measured).** Tracing the receptive field of a single unit near the center of
each named layer's feature map, back to the 32×32 input:

| unit | receptive field | % of the 32×32 input |
|---|---|---|
| one C1 unit | 5×5 | 2.4% |
| one C3 unit | 14×14 | 19.1% |
| one C5 unit | 32×32 | **100.0%** |

![Receptive field growth](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/09_receptive_field_growth.png)

*Which input pixels (red) can affect a single unit at each stage — computed by literally
backpropagating a gradient from that unit to the input, not estimated. A single C1 unit is
genuinely local, seeing only its own 5×5 patch. By C5, a single unit's receptive field
**covers the entire input image** — which is exactly why C5 can behave as a meaningful
"summary" of the whole digit rather than of one local patch, and why
[section 2](#2-the-architecture-laid-out-end-to-end) noted that C5 functions like a dense
layer: with a receptive field equal to the whole input, there is no remaining sense in which it
is doing anything local.*

The theoretical formula matches this exactly: each convolution grows the receptive field by
$(f-1)$, each pool multiplies it by the pool factor $f$:

$$1 \xrightarrow{\text{C1}} 5 \xrightarrow{\text{S2}} 10 \xrightarrow{\text{C3}} 14 \xrightarrow{\text{S4}} 28 \xrightarrow{\text{C5}} 32$$

![Receptive field formula](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/10_receptive_field_formula.png)

*The formula's prediction (line) against the measured receptive field from actual gradient
tracing (boxes) at C1, C3, and C5 — exact agreement at all three: 5, 14, 32. Pooling layers
contribute disproportionately to this growth despite having zero parameters: each 2×2 pool
**doubles** the receptive field's extent in input-pixel terms, which is why S4 alone takes the
field from 14 to 28 — the single largest jump in the whole network.*

![Summary of measured results](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/11_summary_table.png)

*Every row above is a number produced by running the code in this document.*

---

## 8. Implementation from scratch

The two layer types this document adds to what
[convolution-operation.md](convolution-operation.md) and
[pooling-operation.md](pooling-operation.md) already implement — a dense layer and tanh —
plus the network assembled from all of them:

```python
import numpy as np

class Dense:
    def __init__(self, n_in, n_out, seed=0):
        r = np.random.default_rng(seed)
        self.W = r.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out))
        self.b = np.zeros(n_out)

    def forward(self, X):
        self.x = X
        return X @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = dout.sum(0)
        return dout @ self.W.T

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class Tanh:
    def forward(self, X):
        self.out = np.tanh(X)
        return self.out

    def backward(self, dout):
        return dout * (1 - self.out ** 2)     # d/dx tanh(x) = 1 - tanh(x)^2

    def step(self, lr):
        pass


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for l in self.layers:
            X = l.forward(X)
        return X

    def backward(self, dout):
        for l in reversed(self.layers):
            dout = l.backward(dout)
        return dout

    def step(self, lr):
        for l in self.layers:
            l.step(lr)
```

LeNet-5 itself, built from this plus the `Conv2D`, `AvgPool2D`, and `Flatten` layers developed
in [convolution-operation.md](convolution-operation.md) and
[pooling-operation.md](pooling-operation.md):

```python
def build_lenet5(seed=0):
    return Sequential([
        Conv2D(1, 6, 5, seed=seed+0), Tanh(), AvgPool2D(2),     # C1, S2
        Conv2D(6, 16, 5, seed=seed+1), Tanh(), AvgPool2D(2),    # C3, S4
        Conv2D(16, 120, 5, seed=seed+2), Tanh(),                # C5
        Flatten(),
        Dense(120, 84, seed=seed+3), Tanh(),                    # F6
        Dense(84, 10, seed=seed+4),                             # Output (logits)
    ])
```

The framework equivalent, following the Keras conventions the video itself demonstrates:

```python
# Keras
model = keras.Sequential([
    keras.layers.Conv2D(6, 5, activation="tanh", input_shape=(32, 32, 1)),
    keras.layers.AveragePooling2D(2),
    keras.layers.Conv2D(16, 5, activation="tanh"),
    keras.layers.AveragePooling2D(2),
    keras.layers.Conv2D(120, 5, activation="tanh"),
    keras.layers.Flatten(),
    keras.layers.Dense(84, activation="tanh"),
    keras.layers.Dense(10, activation="softmax"),
])
```

Both are drawing the same architecture verified figure-by-figure above; the Keras version is
what almost every modern LeNet-5 tutorial, including the one summarized here, actually builds
— full C3 connectivity, softmax output, no RBF units — rather than a literal reproduction of
the 1998 paper.

---

## 9. Practical guidance

- **Use LeNet-5 as a teaching architecture, not a production one.** At roughly 60,000
  parameters and five learned layers, it is small enough to hold in your head and to train on
  a laptop CPU in seconds — genuinely useful for building intuition, as this document tries to
  demonstrate, but far below the capacity modern vision tasks need.
- **Swap tanh for ReLU and average pooling for max pooling** in any practical reimplementation.
  [Section 5](#5-why-tanh-and-what-it-costs-measured) shows tanh is not strictly worse — it is
  more robust to poor weight scaling — but with correct modern initialization
  ([momentum-optimization.md](momentum-optimization.md), [adam.md](adam.md)) that advantage
  is moot, and ReLU's cheaper computation and typically faster convergence make it the better
  default.
- **Use full C3 connectivity**, not the original paper's hand-designed sparse connection
  table. [Section 4](#4-parameter-count-verified-and-one-surprise) shows this changes the
  parameter count by only a few thousand, and no common framework supports the sparse pattern
  natively — it exists in the original paper purely as an artifact of 1998-era compute
  constraints.
- **Remember that a "5×5 conv, 120 filters" layer can be structurally a dense layer.** C5 is
  the clearest illustration in this document: once a layer's kernel size equals its entire
  spatial input, it has no locality left to exploit, and its 78% share of the network's total
  parameters ([section 4](#4-parameter-count-verified-and-one-surprise)) is the direct
  consequence.
- **Think of receptive field growth as a design budget.** [Section 7](#7-receptive-field-how-far-each-layer-can-see)'s
  measurement — full-image coverage only by C5, the 6th of 7 layers — is a useful sanity check
  for any new architecture: if a task needs global context, make sure enough layers (or
  aggressive-enough pooling/stride) exist for the receptive field to actually reach it.

---

## 10. Key takeaways

1. LeNet-5 alternates convolution and average pooling twice, then finishes with two dense
   layers and a softmax — the pattern essentially every later CNN modifies rather than
   replaces.
2. Every layer's output shape follows the same $n-f+1$ (conv) and divide-by-$f$ (pool)
   formulas from earlier documents in this series, verified here across 9 input sizes with
   zero deviation. The canonical 32×32 input size is not arbitrary: it is exactly what makes
   C5 collapse to a single spatial position.
3. The network has **61,706** learnable parameters under standard full connectivity (the
   commonly cited "≈60,000" reflects the original paper's now-unused sparse C3 connections and
   trainable pooling coefficients). **78% of them sit in a single layer, C5** — despite being a
   convolution by name, it functions as a dense layer once its receptive field equals its
   entire input.
4. The two pooling layers contribute exactly zero parameters, confirmed directly.
5. The tanh-vs-ReLU story is more nuanced than "ReLU is strictly better." With well-scaled
   weights they behave almost identically (gradient shrink ratios of 0.033 vs 0.020). With
   poorly-scaled weights, ReLU's gradient explodes **176× more** than tanh's (10,038× vs 57×),
   because tanh's derivative is capped at 1 and ReLU's is not.
6. A full from-scratch implementation was verified end to end: a real gradient check against
   numerical differentiation matched to 6 significant figures, and the network trained
   successfully (100% test accuracy) on a synthetic stroke-classification task — disclosed
   here as synthetic, since this environment has no access to real MNIST.
7. Receptive field grows fast and is exactly measurable by backpropagating a gradient from a
   single unit to the input: 5×5 (2.4% of the image) at C1, 14×14 (19.1%) at C3, and the
   **entire 32×32 input** by C5 — matching the theoretical $(f{-}1)$-per-conv,
   $\times f$-per-pool formula exactly at every stage tested.

---

## 11. Further reading

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-Based Learning Applied to
  Document Recognition.* Proceedings of the IEEE, 86(11), 2278–2324. — The original LeNet-5
  paper, including the sparse C3 connection table and RBF output layer discussed in
  [sections 2](#2-the-architecture-laid-out-end-to-end) and
  [4](#4-parameter-count-verified-and-one-surprise).
- LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., & Jackel,
  L. D. (1989). *Backpropagation Applied to Handwritten Zip Code Recognition.* Neural
  Computation, 1(4), 541–551. — The direct predecessor architecture, trained on real postal
  zip codes.
- Nair, V., & Hinton, G. E. (2010). *Rectified Linear Units Improve Restricted Boltzmann
  Machines.* ICML. — The paper that popularized ReLU, referenced throughout
  [section 5](#5-why-tanh-and-what-it-costs-measured).
- Glorot, X., & Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward
  neural networks.* AISTATS. — Formal treatment of the vanishing-gradient/saturation problem
  and the weight-initialization schemes that address it, directly relevant to
  [section 5](#5-why-tanh-and-what-it-costs-measured)'s well-scaled-vs-poorly-scaled
  comparison.
- LeCun, Y., Cortes, C., & Burges, C. J. C. *The MNIST Database of Handwritten Digits.*
  http://yann.lecun.com/exdb/mnist/ — The real dataset LeNet-5 was built for; this document
  substitutes a synthetic dataset only because of this environment's lack of internet access,
  as disclosed in [section 6](#6-training-lenet-5-on-a-synthetic-task).
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet Classification with Deep
  Convolutional Neural Networks.* NeurIPS. — AlexNet, the architecture that scaled LeNet-5's
  pattern up by orders of magnitude and switched to ReLU; a natural next read via
  [cnn-architectures.md](cnn-architectures.md).
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, chapter 9.11. MIT
  Press. — Historical notes on convolutional network architectures including LeNet.

---

*Part of a series of deep-learning study notes. Related:
[convolution-operation.md](convolution-operation.md) ·
[pooling-operation.md](pooling-operation.md) ·
[gradient-descent.md](gradient-descent.md) ·
[momentum-optimization.md](momentum-optimization.md) ·
[adam.md](adam.md) ·
[cnn-architectures.md](cnn-architectures.md)*
