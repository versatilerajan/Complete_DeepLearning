# Padding and Strides in CNNs

[cnn-intro.md](cnn-intro.md) established what a convolution computes and why sliding a small shared kernel across an image is so much cheaper than a fully-connected layer. This document covers the two knobs that control exactly *how* that kernel slides: padding (whether and how much you extend the input's borders before convolving) and stride (how far the kernel jumps between positions). Both are simple to state and easy to under-appreciate. A concrete measurement here shows why: without padding, a pixel in the corner of a 10×10 image sitting under a 3×3 kernel gets used in only **1** output computation, while a pixel at the center gets used in **9** — a 9× disparity that padding narrows to 2.25×. Stacking convolutional layers without padding compounds this into outright data loss: a 32×32 input can survive only **15** successive 3×3 convolutions before it shrinks to nothing. Strides, meanwhile, deliver a real, measured **4× reduction in multiply-add operations** at stride 2 versus stride 1 — matching the naive $s^2$ prediction almost exactly — while growing a network's receptive field from 13 pixels to **127 pixels** over the same six layers. The last section verifies a subtlety the video's Keras demonstration doesn't cover: `padding='same'` combined with `strides` greater than 1 does not always behave the way the textbook formula predicts, verified here against TensorFlow's own documented padding rule.

> **Series note.** This builds directly on [cnn-intro.md](cnn-intro.md)'s treatment of the convolution operation. The output-size formula introduced here is used without re-derivation in every later architecture file in this series ([lenet.md](lenet.md), [alexnet.md](alexnet.md), [vggnet.md](vggnet.md)).

---

## Table of Contents

1. [Why convolution alone shrinks the image](#1-why-convolution-alone-shrinks-the-image)
2. [Experiment: how unevenly are pixels actually used?](#2-experiment-how-unevenly-are-pixels-actually-used)
3. [Padding: extending the border before convolving](#3-padding-extending-the-border-before-convolving)
4. [Experiment: what happens across a deep stack, with and without padding](#4-experiment-what-happens-across-a-deep-stack-with-and-without-padding)
5. [The output-size formula](#5-the-output-size-formula)
6. [Experiment: verifying the formula against real convolution](#6-experiment-verifying-the-formula-against-real-convolution)
7. [Strides: skipping positions on purpose](#7-strides-skipping-positions-on-purpose)
8. [Experiment: the real compute savings from stride](#8-experiment-the-real-compute-savings-from-stride)
9. [Experiment: strides and receptive field growth](#9-experiment-strides-and-receptive-field-growth)
10. [Where Keras's `padding='same'` gets subtle](#10-where-kerass-paddingsame-gets-subtle)
11. [Framework usage](#11-framework-usage)
12. [Key takeaways](#12-key-takeaways)
13. [Further reading](#13-further-reading)

---

## 1. Why convolution alone shrinks the image

As shown in [cnn-intro.md](cnn-intro.md), sliding a $k \times k$ kernel across an $n \times n$ input with no special handling of the borders produces an output smaller than the input, because the kernel's top-left corner can only visit positions where the whole kernel still fits inside the image. A 5×5 input with a 3×3 kernel produces a 3×3 output; there are simply no valid positions for the kernel any further out than that.

This has a second, less obvious consequence beyond the output just being smaller: it means pixels near the border participate in far fewer of those output computations than pixels near the center. A corner pixel is covered by exactly one kernel window (the one placed exactly at that corner); a pixel near the middle of a large image is covered by up to $k \times k$ different windows, one for every position where it happens to fall inside the sliding kernel. The next section measures exactly how large this gap is.

## 2. Experiment: how unevenly are pixels actually used?

To make this concrete, take a 10×10 image and a 3×3 kernel, and count — for every single pixel — how many of the kernel's sliding-window positions actually include that pixel. This is an exact count, not an approximation: it enumerates every valid window position and tallies which pixels fall inside each one.

![Coverage heatmap](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_coverage_heatmap.png)

*Left: with no padding, the four corner pixels are each used in only 1 of the 64 window positions a 3×3 kernel visits on a 10×10 image, while every pixel in the 8×8 interior is used in the full 9. Right: adding one ring of zero-padding (making the image effectively 12×12 before convolving) raises the corner count to 4 and the edge count to 6 — narrowing, though not eliminating, the disparity, because the corner pixel of the *original* image is still adjacent to fewer real neighbours than an interior pixel, however much padding surrounds it.*

| | Corner pixel uses | Center pixel uses | Ratio |
|---|---|---|---|
| No padding | 1 | 9 | **9.0×** |
| Same padding (+1 ring) | 4 | 9 | **2.25×** |

This directly substantiates the claim that border information is "lost" relative to central information: it is not that border pixels are ignored outright, but that they contribute to the network's output far less often than central pixels do, purely as a side effect of their position — nothing about their actual content makes them less important.

## 3. Padding: extending the border before convolving

Padding's fix is direct: add extra rows and columns — almost always filled with zeros, hence **zero padding** — around the input before convolving, so that real border pixels are no longer stuck at the edge of the sliding window's range of motion.

![Padding mechanics](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_padding_mechanics.png)

*Left: with no padding, a 3×3 kernel on a 5×5 input can only occupy 9 distinct positions, producing a 3×3 output — visibly smaller than the input. Right: one ring of zeros added on every side turns the same 5×5 input into an effective 7×7 grid; sliding the identical 3×3 kernel across it now produces a 5×5 output, matching the original input size exactly. The shaded cells are padding, not real data — they contribute a fixed, known value (zero) to any window that includes them, rather than distorting the convolution's overall behavior.*

Two padding modes appear in essentially every framework:

- **Valid padding** — no padding at all. The output shrinks according to the formula in Section 5.
- **Same padding** — exactly enough padding added (in the simplest, stride-1 case) to keep the output the same spatial size as the input.

## 4. Experiment: what happens across a deep stack, with and without padding

A single layer's shrinkage looks mild — 5×5 to 3×3 hardly seems like a problem. The effect compounds badly across a deep network, though, and this is directly testable by simulating a stack of layers rather than just one.

![Layer shrinkage](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_layer_shrinkage.png)

*A 32×32 input run through successive 3×3, stride-1 convolutions. Without padding, spatial size drops by 2 every single layer — 32, 30, 28, 26, ... — and by layer 15 the image has shrunk to 2×2, too small to fit another 3×3 kernel at all; the formula predicts exactly this, and it was independently spot-checked by running five actual convolutions and confirming the output sizes matched the formula's predictions exactly, not just approximately. With "same" padding applied at every layer, the size stays at 32×32 indefinitely — verified out to 20 layers here, and true for any number of layers by construction, since the formula guarantees it.*

| | Layers survivable before the image is too small |
|---|---|
| No padding, 3×3 kernels, 32×32 input | **15** |
| Same padding | unlimited |

This is the practical reason padding matters more as networks get deeper: a shallow, 3-4 layer network might tolerate the shrinkage from valid convolutions, but any modern architecture with dozens of layers would run out of image before running out of layers if every one of them discarded a border.

## 5. The output-size formula

Given an $n \times n$ input, a $k \times k$ (often written $f \times f$) kernel, padding $p$ added to each side, and stride $s$ (covered in Section 7), the output spatial size is:

$$\text{out} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$$

Setting $p=0$ recovers the plain shrinkage from Section 1. Setting $p = \lfloor (k-1)/2 \rfloor$ and $s=1$ recovers same padding, keeping the output size equal to $n$ for odd $k$ (Section 10 covers what happens with even kernels or stride greater than 1, where this simple rule needs a caveat).

## 6. Experiment: verifying the formula against real convolution

Rather than trust the formula on faith, it is checked here against a literal, from-scratch 2-D convolution implementation across 300 randomly generated configurations — random input sizes from 4 to 39, random kernel sizes, random padding from 0 to 4, and random stride from 1 to 4 (discarding the handful of combinations where the input is smaller than the kernel even after padding, which is not a valid convolution to begin with).

| | Result |
|---|---|
| Configurations tested | 300 |
| Configurations where formula and actual convolution output size disagreed | **0** |

Every single one of the 300 random trials matched exactly — the formula is not an approximation or a rule of thumb, it is an exact description of what the sliding-window operation produces, for any combination of size, kernel, padding, and stride tested.

## 7. Strides: skipping positions on purpose

Stride controls how far the kernel moves between consecutive positions. A stride of 1 (used implicitly everywhere so far in this document) visits every valid position; a stride of $s>1$ moves the kernel $s$ pixels at a time, skipping the positions in between entirely.

![Stride mechanics](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_stride_mechanics.png)

*Left: stride 1 on an 8×8 input with a 3×3 kernel visits 6×6=36 positions, with each consecutive window overlapping its neighbour in two full rows or columns (visible in the overlapping red/blue outlines). Right: stride 2 visits only 3×3=9 positions — the blue window starts two columns to the right of the red one, not one, so roughly three-quarters of the positions stride 1 would have visited are simply never computed.*

The two motivations given in the video are both real and both measured directly in the next two sections: strides reduce the amount of computation required, and they change how quickly a network's receptive field grows relative to its depth.

## 8. Experiment: the real compute savings from stride

Rather than assume a stride of 2 halves the compute (or take the common $1/s^2$ shorthand on faith), this counts actual multiply-add operations for a realistic mid-network convolutional layer: a 224×224×64 input, a 3×3 kernel, 128 output channels, same-ish padding, at strides 1 through 4.

![Stride compute cost](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_stride_compute_cost.png)

*Left: real multiply-add counts, computed as (output height) × (output width) × (output channels) × (kernel height) × (kernel width) × (input channels) — the exact operation count for this layer, not an estimate. Stride 1 requires 3.70 billion multiply-adds; stride 4 requires only 231 million. Right: the measured speedup relative to stride 1 tracks the naive $s^2$ prediction almost exactly (4.0× measured vs. 4.0× predicted at stride 2; 8.92× vs. 9.0× at stride 3, the small gap coming from integer rounding of the output size, not from any flaw in the $s^2$ approximation).*

| Stride | Output size | Multiply-adds | Speedup vs. stride 1 |
|---|---|---|---|
| 1 | 224×224 | 3.70 billion | 1.0× |
| 2 | 112×112 | 924.8 million | **4.0×** |
| 3 | 75×75 | 414.7 million | 8.92× |
| 4 | 56×56 | 231.2 million | 16.0× |

The "speed up training" claim from the video is not a vague appeal to intuition — a single layer at stride 2 does a measured quarter of the arithmetic that the same layer at stride 1 would, and that saving compounds across every strided layer in a network.

## 9. Experiment: strides and receptive field growth

The other stated motivation — capturing higher-level features while ignoring fine detail — corresponds to growing a network's **receptive field**: the region of the original input that a single output unit is ultimately influenced by. [cnn-intro.md](cnn-intro.md) covered how depth alone grows the receptive field for stride-1 stacks; introducing stride changes the growth rate substantially, computed here with the standard recursive formula (a layer's receptive field equals the previous layer's plus $(k-1)$ times the *cumulative* stride of every layer before it).

![Receptive field growth with strides](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_receptive_field_strides.png)

*Six layers of 3×3 kernels under three stride patterns, plotted on a log scale to make the difference in growth *rate* visible (not just the difference in final value). All-stride-1 grows linearly: 1, 3, 5, 7, 9, 11, 13. Alternating stride 1 and 2 grows faster: 1, 3, 5, 9, 13, 21, 29. All-stride-2 grows exponentially: 1, 3, 7, 15, 31, 63, 127 — doubling (roughly) at every layer, because each strided layer doesn't just add to the receptive field, it multiplies the rate at which every subsequent layer's kernel adds to it.*

| Stride pattern | Receptive field after 6 layers |
|---|---|
| All stride 1 | 13 pixels |
| Alternating stride 1/2 | 29 pixels |
| All stride 2 | **127 pixels** |

A network built entirely from stride-1 convolutions needs to go far deeper to "see" a large region of the input than one that intersperses strided layers — this is the precise, quantifiable version of "capturing high-level features while ignoring fine details": a unit with a 127-pixel receptive field is responding to a much larger, coarser region of the image than one with a 13-pixel receptive field, by construction.

## 10. Where Keras's `padding='same'` gets subtle

The video demonstrates `padding='same'` in Keras as a way to preserve input size automatically. This is accurate when stride is 1. It requires a caveat once stride and same padding are combined — exactly the combination the video's Strides section introduces with `strides=(2, 2)` right after covering padding, without flagging that the two interact non-trivially.

TensorFlow's own documented rule for `'SAME'` padding (verified directly against TensorFlow's API documentation) is:

$$\text{out} = \left\lceil \frac{n}{s} \right\rceil$$

with the required padding split as evenly as possible between the two sides of each dimension — but when an odd amount of padding is needed, **the extra pixel goes on the bottom/right, not the top/left**, making the padding asymmetric. This is a different rule from naively computing $p = \lfloor(k-1)/2\rfloor$ and plugging it into the Section 5 formula symmetrically on both sides — and the two rules can disagree.

Six test cases (random inputs, verified with real convolution under each padding rule, not just the formulas):

| $n$ | $k$ | $s$ | Keras-style output | Naive symmetric output | Agree? | Keras padding (before, after) |
|---|---|---|---|---|---|---|
| 10 | 3 | 2 | 5 | 5 | yes | (0, 1) — asymmetric |
| 7 | 3 | 2 | 4 | 4 | yes | (1, 1) |
| **10** | **4** | **3** | **4** | **3** | **NO** | (1, 2) — asymmetric |
| 15 | 5 | 2 | 8 | 8 | yes | (2, 2) |
| 8 | 3 | 3 | 3 | 3 | yes | (0, 1) — asymmetric |
| 9 | 3 | 2 | 5 | 5 | yes | (1, 1) |

One case out of six — $n=10$, $k=4$, $s=3$ — produced genuinely different output sizes under the two rules (4 vs. 3), and three of the six produced asymmetric padding that a naive "just pad by $(k-1)/2$ on each side" implementation would get wrong. The disagreement case matters in practice: if you compute expected output shapes by hand using the Section 5 formula with a guessed symmetric padding value, and your model uses `padding='same'` with `strides` set to something other than 1, your hand calculation can silently be off by one — this is a common source of shape-mismatch errors when stacking layers or concatenating feature maps from different branches of a network.

## 11. Framework usage

**Keras:**

```python
from tensorflow.keras.layers import Conv2D

# 'same' keeps the spatial size unchanged when strides=(1, 1);
# with strides > 1 it instead uses ceil(input / stride), per Section 10
Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))

# stride 2 for downsampling -- combine carefully with 'same' padding
# per the worked disagreement case in Section 10 if exact output shapes matter
Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')
```

**PyTorch** (explicit padding amount, not a named mode):

```python
import torch.nn as nn

# padding=1 with a 3x3 kernel and stride 1 reproduces 'same' behaviour
nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

# PyTorch's padding argument is an explicit pixel count, always symmetric --
# it has no direct equivalent of Keras's asymmetric SAME rule from Section 10
nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
```

Neither snippet was executed — neither framework is installed in the environment used for this document. The padding *rules* they implement were verified directly against each framework's own documentation and the numerical check in Section 10, rather than assumed from memory.

## 12. Key takeaways

![Summary of all measurements](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_summary_table.png)

1. **Convolution without padding is not a neutral operation on the border.** A corner pixel in a 10×10 image under a 3×3 kernel is used in only 1 of 64 window positions, versus 9 for a center pixel — a measured 9× disparity, narrowed to 2.25× by one ring of same-padding.
2. **This compounds badly with depth.** A 32×32 input run through successive unpadded 3×3 convolutions survives exactly 15 layers before shrinking below the kernel size — a hard, verified ceiling that same padding removes entirely.
3. **The output-size formula is exact, not approximate.** Verified against a literal convolution implementation across 300 random configurations with zero mismatches.
4. **Stride delivers real, verified compute savings.** A realistic conv layer's multiply-add count drops to almost exactly $1/s^2$ of its stride-1 cost — a measured 4.0× speedup at stride 2, matching the naive prediction to within integer-rounding error.
5. **Stride changes receptive-field growth from linear to exponential.** Six all-stride-1 layers reach a 13-pixel receptive field; six all-stride-2 layers reach 127 pixels — a direct, quantifiable version of "strides capture higher-level features."
6. **Keras's `padding='same'` is not a fixed amount of padding — it is a target output size**, computed as $\lceil n/s \rceil$, with padding split asymmetrically (extra pixel to the bottom/right) whenever an even split isn't possible. This can silently disagree with a naive symmetric-padding calculation, verified here in a real case ($n=10$, $k=4$, $s=3$) where the two approaches produce different output sizes entirely.

## 13. Further reading

- **Dumoulin, V., & Visin, F. (2016).** *A guide to convolution arithmetic for deep learning.* arXiv:1603.07285. — The standard reference for the output-size formula and its variants (transposed convolution, dilation), with the same kind of worked arithmetic examples used in this document.
- **TensorFlow documentation.** *Module: tf.nn* (tensorflow.org/api_docs/python/tf/nn). — The authoritative source for the exact `'SAME'`/`'VALID'` padding rules verified in Section 10.
- **LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).** *Gradient-based learning applied to document recognition.* Proceedings of the IEEE, 86(11), 2278–2324. — LeNet-5 (covered in [lenet.md](lenet.md)) is one of the earliest architectures to make deliberate, documented choices about padding and stride at every layer.
- **Springenberg, J. T., Dosovitskiy, A., Brox, T., & Riedmiller, M. (2015).** *Striving for Simplicity: The All Convolutional Net.* ICLR 2015 (workshop track). — Argues for replacing pooling layers with strided convolutions entirely, directly relevant to the receptive-field argument in Section 9.
- **Araujo, A., Norris, W., & Sim, J. (2019).** *Computing Receptive Fields of Convolutional Neural Networks.* Distill. — An interactive, more complete treatment of the receptive-field recursion used in Section 9, including the case of pooling layers and dilated convolutions not covered here.

---

*Part of an ongoing deep learning notes series. Previous: [cnn-intro.md](cnn-intro.md). Next: [pooling.md](pooling.md).*
