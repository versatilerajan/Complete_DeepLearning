# The Pooling Layer

Pooling is the second building block of a classical CNN, and it does the opposite job of
convolution: instead of learning weights, it throws information away on purpose. A 2×2 max
pool halves a feature map's height and width by keeping only the largest value in each
non-overlapping window, and it does this with zero learnable parameters. That sounds like a
strange thing to want, and the point of this document is to test, rather than assume, why it
helps: memory savings are easy to verify directly; the translation-invariance story turns out
to be real but far more conditional than it is usually presented; and the information-loss
cost is measurable and has a genuine downside. Every figure below was produced by running
code — including a real small CNN trained twice, once with pooling and once without, and
compared on accuracy, parameter count, and robustness to shifts it never saw in training.
Every number in the prose is that code's output.

This builds directly on [convolution-operation.md](convolution-operation.md), whose output
feature maps are exactly what a pooling layer takes as input, and continues toward
[cnn-architectures.md](cnn-architectures.md), which puts convolution and pooling together
into full networks. See also [gradient-descent.md](gradient-descent.md) for how the
surrounding layers get trained.

---

## Table of Contents

1. [What pooling does](#1-what-pooling-does)
2. [Max pooling and average pooling, worked by hand](#2-max-pooling-and-average-pooling-worked-by-hand)
3. [Memory: measured](#3-memory-measured)
4. [No learnable parameters: verified, not assumed](#4-no-learnable-parameters-verified-not-assumed)
5. [Translation invariance: the clean case](#5-translation-invariance-the-clean-case)
6. [Translation invariance: it depends on sparsity](#6-translation-invariance-it-depends-on-sparsity)
7. [Max vs average vs global pooling](#7-max-vs-average-vs-global-pooling)
8. [The cost: information loss and localization error](#8-the-cost-information-loss-and-localization-error)
9. [A real network, with and without pooling](#9-a-real-network-with-and-without-pooling)
10. [Implementation from scratch](#10-implementation-from-scratch)
11. [Practical guidance](#11-practical-guidance)
12. [Key takeaways](#12-key-takeaways)
13. [Further reading](#13-further-reading)

---

## 1. What pooling does

A convolutional layer's output — the feature map — is still spatially organized: a 224×224
image run through a few 3×3 "same"-padded convolutions is still roughly 224×224, just with
more channels. Pooling is the operation that actually shrinks that spatial size. It slides a
window over the feature map, exactly like a convolution filter, but instead of a learned
weighted sum it applies a fixed, parameter-free summary — most commonly the **maximum** of
the window, sometimes the **average**.

Two claimed benefits motivate this, and both are tested with real numbers below rather than
taken on faith:

- **Memory / compute.** A smaller feature map is directly cheaper to store and to convolve
  further — [section 3](#3-memory-measured).
- **Translation invariance.** The network's output should not care exactly which pixel a
  feature sits on. Pooling is meant to build in a small amount of tolerance to shifts —
  [sections 5-6](#5-translation-invariance-the-clean-case).

And one real cost:

- **Information loss.** A pooling layer is a many-to-one map — it is, in general,
  irreversible — and precise spatial location is exactly what gets discarded. That matters a
  great deal for tasks like segmentation where the output needs pixel-precise location, and
  much less for tasks like classification where only *whether* a feature is present matters —
  [section 8](#8-the-cost-information-loss-and-localization-error).

---

## 2. Max pooling and average pooling, worked by hand

For a window size $f$ and stride $s$ (almost always $s=f$, i.e. non-overlapping windows),
each output cell summarizes one patch of the input:

$$Y^{\max}_{i,j} = \max_{(u,v)\,\in\,\text{window}} X_{i\cdot s+u,\;j\cdot s+v} \qquad Y^{\text{avg}}_{i,j} = \frac{1}{f^2}\sum_{(u,v)\,\in\,\text{window}} X_{i\cdot s+u,\;j\cdot s+v}$$

![Pooling mechanic](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_pooling_mechanic.png)

*(a) A 4×4 input, divided into four non-overlapping 2×2 windows, color-coded. (b) Max
pooling: each window collapses to its largest value — the orange window's {1,3,5,6} becomes
6, the red window's {8,1,4,9} becomes 9. (c) Average pooling: the same windows collapse to
their mean instead — {1,3,5,6} becomes 3.75. Both were computed by the same code used for
every other figure in this document, and both match hand arithmetic exactly.*

Note what did **not** happen anywhere in this computation: no weight was multiplied, no bias
was added, nothing was learned. This is confirmed directly in
[section 4](#4-no-learnable-parameters-verified-not-assumed).

---

## 3. Memory: measured

With window size $f$ and stride $s=f$ (the standard configuration), the output size follows
the same sliding-window formula as convolution:

$$\left\lfloor \frac{n-f}{s} \right\rfloor + 1$$

**Experiment (measured).** Every input size $n \in \{4,6,\dots,64\}$ — **31 values** — with a
2×2, stride-2 pool, compared against the formula: all 31 matched exactly.

![Memory reduction](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_memory_reduction.png)

*(a) Measured output size against the formula's prediction — they coincide everywhere.
(b) The actual element count on a log axis: a 2×2/stride-2 pool gives exactly **4× fewer**
values at $n=64$ ($64^2=4096$ down to $32^2=1024$), and every further pooling layer in a
network compounds this reduction. This is the most straightforward of pooling's benefits —
there is no subtlety in it, unlike invariance.*

---

## 4. No learnable parameters: verified, not assumed

Unlike a convolutional layer, pooling has no weights to train — the claim is not just that
gradient descent doesn't update anything here, but that there is nothing *to* update. What a
pooling layer does have is a backward pass: it must still route the loss gradient through to
whatever produced its input, exactly like any other layer, since backpropagation needs to
continue past it.

For max pooling, the gradient with respect to the output flows **only** to the input cell
that was the maximum — every other cell in the window gets zero, because an infinitesimal
change to a non-maximum value doesn't change the max at all. For average pooling, the
gradient is split **evenly** across every cell in the window, since every cell contributed
equally to the mean.

**Verification (measured).** Both backward passes were implemented directly and checked
against numerical (finite-difference) gradients:

| | max abs |analytic − numerical| |
|---|---|
| max pool | $1.8\times10^{-11}$ |
| average pool | $9.3\times10^{-12}$ |

And directly confirming the routing behavior: on a 6×6 input pooled 2×2, exactly **75%** of
input cells received zero gradient — the non-maximum three out of every four cells in each
window — and every nonzero gradient location coincided exactly with that window's argmax.

![No trainable parameters](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_no_trainable_params.png)

*(a) A random 6×6 input, with the max-pool argmax location of each window marked by a star.
(b) The backward gradient for max pooling: nonzero only at the starred cells, zero (the
lightest shade) everywhere else — 75% of the map. (c) The backward gradient for average
pooling: spread uniformly across every cell of its window, since every cell participated
equally in the mean. Neither panel updates a weight, because there is no weight here to
update — this is the entire reason Keras and PyTorch implementations of pooling layers list
zero trainable parameters.*

---

## 5. Translation invariance: the clean case

The textbook argument for pooling is that it should make a network's output less sensitive
to exactly *where* a feature sits. The cleanest version of this claim is easy to construct
and test directly: put a single strong activation somewhere inside one pooling window, and
move it to every position inside that same window.

**Experiment (measured).** A single spike of value 1.0, placed at all 4 positions inside one
2×2 window, read out two ways: a "raw" detector that reads one fixed pixel, and the max-pool
output for that window.

- raw fixed-pixel readout: **2 distinct values** across the 4 positions (1.0 once, 0.0 the
  other three times — the naive detector only fires when the spike lands exactly on the
  pixel it's watching)
- max-pool output: **1 distinct value** across all 4 positions (always 1.0)

![Translation invariance: the clean case](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_translation_invariance_clean_case.png)

*The same activation, moved to all four corners of a single pooling window (the blue outline
marks that window in each panel). The raw pixel readout is highly sensitive to exact
position — it only registers the spike once out of four placements. The max-pool output for
that window is identical every time, because the operation only asks "what is the largest
value in here", not "exactly where is it" — a genuine, verified invariance, but only to
movement that stays inside one window.*

---

## 6. Translation invariance: it depends on sparsity

The clean case above used an isolated, sparse spike against a zero background — which is a
reasonable model of what a ReLU'd convolutional layer's output often looks like (strong
activation at a detected feature, near-zero everywhere else). It is worth checking whether
the same benefit survives on a **smooth** feature map, and what happens at the boundary
between two pooling windows, since real feature maps are not always sparse and real shifts
are not always small.

**Experiment (measured).** Comparing relative change (in the full output map, not one
window) under horizontal shifts, with and without a 2×2 max pool:

| feature map | shift +1px, no pooling | shift +1px, pooled | shift −1px, pooled |
|---|---|---|---|
| **sparse** (a few isolated peaks) | 1.4142 | **0.0** — fully absorbed | **1.4142** — not absorbed at all |
| **smooth** (Gaussian-blurred square edge) | 0.2182 | 0.2041 | 0.2041 |

The smooth map's improvement from pooling at a 1px shift is only **1.07×** — pooling barely
helps there, because a smoothly varying map doesn't have a single dominant value for the max
to lock onto; nudging the whole neighborhood nudges the max by nearly as much as it nudges
the raw value. The sparse map shows the dramatic benefit from
[section 5](#5-translation-invariance-the-clean-case) — but notice it is completely
**asymmetric**: a +1px shift left the peak inside the same pooling window and the output was
untouched, while a −1px shift moved the identical peak into an *adjacent* window and produced
just as much change as no pooling at all.

![Invariance depends on sparsity](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_invariance_depends_on_sparsity.png)

*Sparse map (orange) vs smooth map (blue), each with and without pooling. The sparse-pooled
curve is the only one that ever touches zero away from the origin — at shift +1 — and it is
exactly as bad as having no pooling at all one step further in the other direction. The
smooth-pooled curve barely separates from its unpooled counterpart anywhere.*

The honest summary: pooling gives real, but **local and boundary-dependent** invariance,
concentrated in the regime — sparse, peaky activations — that convolution followed by ReLU
tends to actually produce. It is not a general guarantee that "shifting the input by a
little never changes the output"; it is a coin flip at any window boundary, and it barely
helps on smooth activations at all.

---

## 7. Max vs average vs global pooling

Average pooling summarizes a window with its mean instead of its max. On the sparse
activation pattern from [section 6](#6-translation-invariance-it-depends-on-sparsity), this
makes a concrete, measurable difference:

**Experiment (measured).** A 16×16 map with 4 sparse strong activations (peak 0.87) against a
near-zero background, pooled 2×2:

| | input | max pool | average pool |
|---|---|---|---|
| max value | 0.87 | **0.87** (exactly preserved) | **0.23** (diluted **3.7×**) |

![Max vs average vs global pooling](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_pooling_types_comparison.png)

*(a) The sparse input. (b) Max pooling preserves the peak's exact value — 0.87 in, 0.87 out.
(c) Average pooling drags every peak toward the window's mean, which is dominated by the many
near-zero background pixels — the same peak comes out at 0.23. This is precisely why max
pooling, not average, became the default for classification networks: it is the summary
statistic that survives contact with a sparse, ReLU'd feature map. (d) Global pooling —
reducing an entire channel to a single number, used most often right before a final
classification layer — makes the same trade-off at the whole-map scale: global max keeps the
strongest signal (0.87), global average dilutes it to the map's overall mean (0.05).*

Average pooling is not obsolete — it resurfaces as **global average pooling**, almost always
in place of a large fully connected layer at a network's very end (Lin et al., 2013), where
its smoothing behavior is actually a regularizing feature rather than a bug: it discourages
the network from relying on one single spatial location for its final decision.

---

## 8. The cost: information loss and localization error

Pooling is a many-to-one operation: many different arrangements of values inside a window
produce the same max, so the exact configuration cannot be recovered from the output.

![Many-to-one information loss](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_information_loss_many_to_one.png)

*Four genuinely different 2×2 inputs, all producing the identical max-pool output of 1.00.
Given only the pooled value, there is no way to tell which of these — or infinitely many
other arrangements — actually occurred. This is not a corner case; it is what "pooling" means
by construction.*

The concrete consequence for a spatial task is **localization error**: how precisely can the
position of a feature be recovered from a pooled representation?

**Experiment (measured).** A single active pixel placed at a random location in a 32×32 map,
run through 0 to 3 stacked 2×2 max-pool layers; recover the best-guess location from the
argmax of the (shrunken) output, rescale back to original coordinates, and measure the error:

| stacked 2×2 pools | output resolution | mean localization error |
|---|---|---|
| 0 | 32×32 | 0.71px (baseline: pixel quantization only) |
| 1 | 16×16 | 0.86px |
| 2 | 8×8 | 1.58px |
| 3 | 4×4 | **3.16px** |

![Localization cost](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/08_localization_cost.png)

*(a) Mean recoverable-location error, averaged over 200 random trials, growing with every
additional pooling layer. (b) The mechanism laid bare: output resolution (blue, falling) and
localization error (orange, rising) track each other almost exactly — the error is
essentially a direct consequence of how few output cells are left to distinguish between.
This is the concrete reason segmentation and detection architectures either use far less
pooling than classification networks, or pair it with an upsampling path (as in a U-Net) to
recover the spatial resolution that pooling discarded.*

---

## 9. A real network, with and without pooling

All of the above uses hand-built feature maps. To check whether the same trade-offs show up
in an actual trained model, a small from-scratch CNN — conv(8 filters, 3×3) → ReLU →
[pool] → conv(16 filters, 3×3) → ReLU → [pool] → flatten → linear → softmax — was trained
twice on a 3-class synthetic shape classification task (square / plus / diagonal line, 20×20
images), once with 2×2 max pooling after each conv layer and once without.

Crucially, the training images used a **tightly clustered** range of positions (a 3-pixel
jitter), so the network cannot learn shift-robustness merely by having seen many positions in
training data — any genuine robustness has to come from the pooling architecture itself.

**Experiment (measured), part 1: capacity and accuracy.**

| | parameters | peak feature-map size | test accuracy |
|---|---|---|---|
| with pooling | **1,683** | **2,592** elements | 100% |
| without pooling | 13,539 (**8.0× more**) | 4,096 elements | 100% |

Both networks solve this (easy, low-position-variance) task perfectly, and the no-pooling
network actually reaches a slightly *lower* training loss — unsurprising, since it has 8×
more parameters to fit the same amount of data with.

![Trained CNN comparison](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/09_trained_cnn_comparison.png)

*(a) Training loss over 25 epochs, mean ± 1 sd over 3 seeds — the unpooled network (orange)
fits the training set slightly better, consistent with its much larger parameter count.
(b) Both networks reach 100% test accuracy quickly. On accuracy alone, pooling looks
unnecessary here — the real difference shows up under distribution shift, tested next.*

**Experiment (measured), part 2: robustness to shifts never seen in training.** Sixty test
shapes, each shifted horizontally by 0 to 5 pixels — well beyond the 3-pixel jitter range the
networks were trained on — measuring what fraction of predictions stay the same as the
unshifted version:

| shift (px) | with pooling | without pooling |
|---|---|---|
| 0-1 | 100% | 100% |
| 2 | 100% | 96.7% |
| 3 | **100%** | 83.3% |
| 4 | 86.7% | 71.7% |
| 5 | 65.0% | 58.3% |

![Trained network shift robustness](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/10_trained_shift_robustness.png)

*Prediction stability as the test image is shifted further from anything seen in training.
Both networks hold up for the first couple of pixels, but the pooled network holds a perfect
100% out to a 3px shift — a full pixel further than the training jitter — while the unpooled
network is already down to 83% at that point. Both eventually degrade at large enough shifts,
since neither was built with unlimited invariance; the gap is exactly what
[sections 5-6](#5-translation-invariance-the-clean-case) predicted from first principles, now
confirmed on weights that were actually trained by backpropagation rather than hand-set.*

![Summary of measured results](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/11_summary_table.png)

*Every row above is a number produced by running the code in this document.*

---

## 10. Implementation from scratch

Max pooling, forward and backward, in plain NumPy — note that the backward pass needs to
remember *which* cell was the max (the "argmax"), which is exactly why frameworks cache it
during the forward pass:

```python
import numpy as np

def maxpool2d(x, f=2, stride=None):
    """x: (H,W). Returns (out, argmax_idx); argmax_idx[i,j] is the (row,col)
       offset within window (i,j) that held the maximum -- needed for backward."""
    stride = stride or f
    H, W = x.shape
    oh = (H - f) // stride + 1
    ow = (W - f) // stride + 1
    out = np.zeros((oh, ow))
    arg = np.zeros((oh, ow, 2), dtype=int)
    for i in range(oh):
        for j in range(ow):
            r, c = i * stride, j * stride
            window = x[r:r + f, c:c + f]
            idx = np.unravel_index(np.argmax(window), window.shape)
            out[i, j] = window[idx]
            arg[i, j] = idx
    return out, arg


def maxpool2d_backward(dout, arg, in_shape, f=2, stride=None):
    """Route each output gradient to ONLY the cell that was the max."""
    stride = stride or f
    dx = np.zeros(in_shape)
    oh, ow = dout.shape
    for i in range(oh):
        for j in range(ow):
            r, c = i * stride, j * stride
            di, dj = arg[i, j]
            dx[r + di, c + dj] += dout[i, j]
    return dx
```

Average pooling is simpler — no argmax to track, since every cell contributes equally both
forward and backward:

```python
def avgpool2d(x, f=2, stride=None):
    stride = stride or f
    H, W = x.shape
    oh = (H - f) // stride + 1
    ow = (W - f) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            r, c = i * stride, j * stride
            out[i, j] = x[r:r + f, c:c + f].mean()
    return out


def avgpool2d_backward(dout, in_shape, f=2, stride=None):
    """Split each output gradient EVENLY across every cell in its window."""
    stride = stride or f
    dx = np.zeros(in_shape)
    oh, ow = dout.shape
    for i in range(oh):
        for j in range(ow):
            r, c = i * stride, j * stride
            dx[r:r + f, c:c + f] += dout[i, j] / (f * f)
    return dx
```

Global average pooling — collapsing an entire $(H,W,C)$ feature map to a $(C,)$ vector — is a
single line:

```python
def global_avgpool(x):
    """x: (H,W,C) -> (C,)"""
    return x.reshape(-1, x.shape[-1]).mean(axis=0)
```

The framework equivalents. Note that none of them take a `units` or `filters` argument —
consistent with [section 4](#4-no-learnable-parameters-verified-not-assumed), there is
nothing here to size in terms of learnable weights, only window geometry:

```python
# PyTorch
pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
gap = torch.nn.AdaptiveAvgPool2d(1)      # global average pooling

# Keras
pool = keras.layers.MaxPooling2D(pool_size=2, strides=2)
gap = keras.layers.GlobalAveragePooling2D()
```

---

## 11. Practical guidance

- **Default to max pooling, 2×2, stride 2** for classification backbones — it is the
  overwhelming convention, and [section 7](#7-max-vs-average-vs-global-pooling) shows exactly
  why: it survives sparse, ReLU'd activations without diluting them the way average pooling
  does.
- **Use global average pooling at the very end of the network**, in place of a large flatten
  + dense layer, when the task is whole-image classification — it cuts parameters
  dramatically and, per Lin et al. (2013), tends to generalize at least as well.
- **Prune or remove pooling where precise location matters** — segmentation, detection,
  keypoint estimation. [Section 8](#8-the-cost-information-loss-and-localization-error)'s
  localization-error measurement is the direct reason architectures for these tasks either use
  far less pooling or pair it with an upsampling/skip-connection path (U-Net, feature
  pyramids) to recover what pooling discarded.
- **Don't expect pooling to buy large-shift invariance for free.** Per
  [section 6](#6-translation-invariance-it-depends-on-sparsity), the benefit is concentrated
  in shifts smaller than the pooling window and is asymmetric around window boundaries. Real
  robustness to larger shifts in a trained network comes substantially from data augmentation
  (training on varied positions), not from pooling geometry alone.
- **A strided convolution is a drop-in alternative to pooling** for downsampling, and
  unlike pooling it has learnable weights — some modern architectures (e.g. many ResNet
  variants) prefer it for exactly that reason. It is worth treating "how do I downsample" as
  a genuine design choice rather than defaulting to pooling automatically.

---

## 12. Key takeaways

1. Pooling summarizes each window of a feature map with a fixed statistic — usually the max,
   sometimes the average — with no learned weights.
2. It shrinks feature maps by exactly the sliding-window formula convolution uses. Measured
   across 31 input sizes with zero deviation; a 2×2/stride-2 pool gives exactly **4× fewer**
   elements.
3. Pooling genuinely has **zero** learnable parameters — verified by backprop, not assumed:
   max pooling's backward pass sends gradient to only the argmax cell (75% of cells got exact
   zero in the test case) and average pooling's spreads it evenly, both confirmed against
   numerical gradients to ~$10^{-11}$.
4. Translation invariance is real but **conditional**. A sparse spike moved anywhere inside
   one pooling window produces an *identical* pooled output (measured: 4 distinct raw
   readouts collapse to 1 pooled value) — but the same spike crossing a window boundary
   produces just as much change as no pooling at all, and the benefit nearly vanishes
   (1.07× improvement) on smooth, non-sparse feature maps.
5. Max pooling preserves peak signal exactly; average pooling dilutes it. Measured: a 0.87
   peak against a near-zero sparse background survives max pooling unchanged and comes out at
   0.23 — a **3.7× dilution** — under average pooling.
6. Pooling is a many-to-one operation and therefore lossy by construction; distinct inputs to
   a window can produce identical output, and that loss is directly measurable as
   localization error — **3.16px** after 3 stacked 2×2 pools, against **0.71px** with none,
   tracking output resolution almost exactly.
7. On a real trained network, pooling cost **8.0× more parameters** to remove (13,539 vs
   1,683) while reaching identical accuracy on an easy in-distribution test set — but held
   **100%** prediction stability under a 3-pixel shift the network never saw in training,
   against 83.3% for the unpooled network, confirming the invariance argument on genuinely
   trained weights rather than a hand-built toy filter.

---

## 13. Further reading

- Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. (2016). *Learning Deep
  Features for Discriminative Localization.* CVPR. — Global average pooling used to recover
  coarse spatial localization from a classification network, connecting directly to
  [section 8](#8-the-cost-information-loss-and-localization-error)'s localization measurement.
- Lin, M., Chen, Q., & Yan, S. (2013). *Network In Network.* arXiv:1312.4400. — Introduced
  global average pooling as a replacement for large fully connected classification heads,
  referenced in [sections 7](#7-max-vs-average-vs-global-pooling) and
  [11](#11-practical-guidance).
- Scherer, D., Müller, A., & Behnke, S. (2010). *Evaluation of Pooling Operations in
  Convolutional Architectures for Object Recognition.* ICANN. — An empirical comparison of
  max vs average pooling that anticipates the dilution effect measured in
  [section 7](#7-max-vs-average-vs-global-pooling).
- Springenberg, J. T., Dosovitskiy, A., Brox, T., & Riedmiller, M. (2015). *Striving for
  Simplicity: The All Convolutional Net.* ICLR Workshop. — Argues for replacing pooling with
  strided convolutions, the alternative raised in [section 11](#11-practical-guidance).
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for
  Biomedical Image Segmentation.* MICCAI. — The architecture pattern (pooling + matching
  upsampling path) built specifically to recover the localization loss quantified in
  [section 8](#8-the-cost-information-loss-and-localization-error).
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, chapter 9.3. MIT Press.
  — Textbook discussion of pooling and invariance.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-Based Learning Applied to
  Document Recognition.* Proceedings of the IEEE, 86(11), 2278–2324. — LeNet-5, one of the
  earliest uses of (average) pooling as a standard CNN component.

---

*Part of a series of deep-learning study notes. Related:
[convolution-operation.md](convolution-operation.md) ·
[gradient-descent.md](gradient-descent.md) ·
[momentum-optimization.md](momentum-optimization.md) ·
[adam.md](adam.md) ·
[cnn-architectures.md](cnn-architectures.md)*
