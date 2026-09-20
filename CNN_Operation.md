# The Convolution Operation

A convolutional neural network is a normal neural network with one operation swapped in for
matrix multiplication: instead of connecting every input to every output, a small filter
slides across the image and produces one output value per position. That single change —
local, shared weights instead of a full connection — is what lets a network look at a
million-pixel image without a million-times-a-million weight matrix, and it is the entire
subject of this document. Every figure below was produced by running code: real filters
convolved with real (synthetic but exact) images, a from-scratch backward pass checked
against numerical gradients, and a filter recovered from random noise by gradient descent
alone. Every number quoted in the prose is output from that code.

This assumes the material in [gradient-descent.md](gradient-descent.md) and builds toward
[momentum-optimization.md](momentum-optimization.md) and [adam.md](adam.md), which cover how
a network's weights — filters included — actually get updated during training. It feeds into
[pooling-operation.md](pooling-operation.md) and [cnn-architectures.md](cnn-architectures.md).

---

## Table of Contents

1. [Why a specialized operation for images](#1-why-a-specialized-operation-for-images)
2. [How images are stored](#2-how-images-are-stored)
3. [The convolution operation, worked by hand](#3-the-convolution-operation-worked-by-hand)
4. [Convolution as edge detection](#4-convolution-as-edge-detection)
5. [Output size: the n − f + 1 formula](#5-output-size-the-n--f--1-formula)
6. [Padding](#6-padding)
7. [Stride](#7-stride)
8. [Convolving a multi-channel image](#8-convolving-a-multi-channel-image)
9. [Multiple filters: the output volume](#9-multiple-filters-the-output-volume)
10. [Filters are learned, not designed](#10-filters-are-learned-not-designed)
11. [From edges to corners: why depth matters](#11-from-edges-to-corners-why-depth-matters)
12. [Implementation from scratch](#12-implementation-from-scratch)
13. [Practical guidance](#13-practical-guidance)
14. [Key takeaways](#14-key-takeaways)
15. [Further reading](#15-further-reading)

---

## 1. Why a specialized operation for images

A fully connected layer treats every input pixel as independent: a 224×224 RGB image is
150,528 numbers, and a single fully connected layer with 1,000 output units would need over
150 million weights before the network has done anything at all. Two properties of images
make this wasteful, and convolution is built to exploit both:

- **Locality.** Whether a pixel is part of an edge depends on the handful of pixels around
  it, not on a pixel in the opposite corner of the image. A detector for "edge here" only
  needs a small local receptive field.
- **Translation invariance.** An edge detector that works in the top-left corner should work
  unchanged in the bottom-right corner — the same feature can appear anywhere. So the same
  small set of weights can be *reused* at every position instead of learned separately for
  each one.

A convolutional layer encodes both assumptions directly: one small filter (a handful of
weights), applied at every position, producing one number per position. The design was
originally motivated by studies of the mammalian visual cortex, where individual neurons
respond only to a small region of the visual field (a receptive field) and are organized so
that simple responses in early layers — edges, orientation, contrast — combine into
increasingly complex responses in later ones (Hubel & Wiesel, 1962). CNNs reproduce that
structure computationally: early layers detect primitive features like edges, and deeper
layers combine those into corners, textures, parts, and eventually whole objects. Section
[11](#11-from-edges-to-corners-why-depth-matters) builds a small, concrete instance of that
hierarchy from scratch, all the way from a raw image to a working corner detector.

---

## 2. How images are stored

A digital image is a matrix (or a stack of matrices) of numbers. A **grayscale** image is a
single 2D array of intensity values, conventionally integers from 0 (black) to 255 (white).
A **color (RGB)** image is three such arrays stacked together — one per channel — so a
$H \times W$ color image has shape $H \times W \times 3$.

![Image representation](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_image_representation.png)

*(a) An 8×8 grayscale image, shown with its actual stored integers — a bright square (255)
against a dark background (0). This is the entire representation; there is no additional
structure. (b) The same idea in color: three 8×8 planes, one per channel, drawn stacked to
show that "RGB image" literally means three grayscale images glued together along a new
axis. (c) The three channels composited into the color image a viewer would see. Whatever
operation a network performs on a grayscale image, it can perform on an RGB image by simply
having a third dimension to sum over — which is exactly what
[section 8](#8-convolving-a-multi-channel-image) does.*

---

## 3. The convolution operation, worked by hand

Take a small filter (also called a *kernel*) — a matrix of weights, most often 3×3 — and
slide it across the input. At each position, take the element-wise product of the filter
and the patch of the image underneath it, and sum the result into a single number. Repeat
at every position and the collection of outputs forms a new matrix: the **feature map**.

![The sliding-window mechanic](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_sliding_window_mechanic.png)

*(a) A 6×6 input. (b) A 3×3 kernel. (c) The resulting 4×4 output — one value for every
position the 3×3 window can occupy without running off the edge of a 6×6 input, which is
exactly $6-3+1=4$ positions per side ([section 5](#5-output-size-the-n--f--1-formula)). The
highlighted window in (a) and its corresponding output cell in (c) show one step of the
slide: multiply element-wise, then sum.*

Concretely, for the highlighted window:

![Worked arithmetic](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_worked_arithmetic.png)

*The nine element-wise products, summed. This value, computed by hand, was checked against
the from-scratch `conv2d()` function used for every other figure in this document — both
give exactly $-2$.*

In general, for an input $X$ and kernel $K$ of size $f \times f$, the output at position
$(i,j)$ is

$$Y_{i,j} = \sum_{u=0}^{f-1}\sum_{v=0}^{f-1} X_{i+u,\,j+v}\; K_{u,v}$$

One terminology note worth flagging: this operation — no kernel flipping — is what deep
learning libraries call "convolution." It is technically *cross-correlation*; true
mathematical convolution flips the kernel 180° first. The distinction is invisible in
practice because the kernel's weights are learned rather than designed
([section 10](#10-filters-are-learned-not-designed)), so a network trained with either
convention learns whichever kernel orientation its convention needs.

---

## 4. Convolution as edge detection

A fixed, hand-designed kernel can already do useful work. The Sobel operator is a classic
example — two 3×3 kernels, one sensitive to vertical intensity changes and one to
horizontal ones:

$$K_x = \begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix} \qquad K_y = \begin{bmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{bmatrix}$$

**Experiment (measured).** Convolving $K_x$ with a bright square on a dark, slightly noisy
background:

| location | Sobel-X response |
|---|---|
| flat background (no edge) | 0.0 (matches exactly) |
| flat interior of the square (no edge) | ≈ 0.0 (within noise) |
| left edge of the square (dark → bright) | **+3.9** |
| right edge of the square (bright → dark) | **−3.9** |

![Edge detection](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_edge_detection.png)

*(a) The input — a bright square with a small amount of pixel noise added, so the filters
are tested on something more realistic than a perfectly clean image. (b) $K_x$'s response:
strongly positive at the left edge, strongly negative at the right edge, and near zero
everywhere else — exactly what "detects vertical edges, with sign indicating direction" means
in practice. (c) $K_y$, the same behavior rotated 90°. (d) The gradient magnitude
$\sqrt{g_x^2+g_y^2}$, which traces the full outline regardless of edge direction. This is a
real, runnable edge detector; the only thing separating it from a trained CNN's first layer
is that here the weights were chosen by a human instead of learned.*

---

## 5. Output size: the n − f + 1 formula

An $n \times n$ input convolved with an $f \times f$ filter (stride 1, no padding) produces
an output of size

$$(n - f + 1) \times (n - f + 1)$$

The reasoning is direct: the filter's top-left corner can sit at any of $n - f + 1$ positions
along each axis before the filter would extend past the edge of the input.

**Experiment (measured).** Every combination of input size $n \in \{5, \dots, 29\}$ and
filter size $f \in \{2,3,4,5,7\}$ with $f \le n$ — **123 pairs** — computed by running the
actual `conv2d()` sliding-window code and compared against the formula. All 123 matched
exactly; maximum deviation **0**.

![Output size formula](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_output_size_formula.png)

*Every marker is a measured output size from actually running convolution; every line is the
formula's prediction. They coincide everywhere they were tested.*

Two consequences worth internalizing early, because both come up constantly in practice.
First, the output always **shrinks** relative to the input (for $f>1$) — a problem addressed
in [section 6](#6-padding). Second, a $1\times1$ filter ($f=1$) leaves the spatial size
completely unchanged and reduces to a per-pixel weighted combination across channels — a
common trick for adjusting depth without touching height or width.

---

## 6. Padding

Shrinking on every layer is a problem for two reasons: it limits how deep a network can go
before running out of spatial dimensions, and it discards information disproportionately
from the borders, since a pixel at the edge of the image participates in far fewer output
computations than a pixel near the center.

**Padding** adds a border of (typically zero-valued) pixels around the input before
convolving, so the output can be made to match the input size. For a filter of size $f$
(odd), the padding that keeps the output the same size as the input — called **"same"**
padding — is

$$p = \frac{f-1}{2}$$

**Experiment (measured).** A 10×10 input with a 3×3 filter:

| padding mode | padding added | output size |
|---|---|---|
| "valid" (none) | 0 | **8×8** |
| "same" | $(3-1)/2 = 1$ | **10×10** — matches the formula exactly |

Stacking layers compounds the shrinkage. Five successive 3×3 "valid" convolutions on a
64×64 image:

| after $k$ layers | "valid" size | area remaining | "same" size | area remaining |
|---|---|---|---|---|
| 0 | 64×64 | 100% | 64×64 | 100% |
| 5 | **54×54** | **71.2%** | 64×64 | 100% |

![Padding](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_padding.png)

*(a), (b) A 10×10 input and its "valid" convolution output, visibly smaller and missing a
one-pixel border all the way around. (c) The measured area remaining after stacking up to
five 3×3 convolutions: "valid" (orange) loses ground steadily — down to 71.2% of the
original area after five layers — while "same" (blue) holds flat at 100% by construction. A
real CNN of any depth almost always uses "same" padding for exactly this reason.*

---

## 7. Stride

Stride is the step size the filter moves between applications. Stride 1 (used everywhere
above) slides one pixel at a time; a larger stride skips positions, producing a smaller,
coarser output — a cheap way to downsample. The general output-size formula, combining
padding $p$, filter size $f$, and stride $s$:

$$\left\lfloor \frac{n + 2p - f}{s} \right\rfloor + 1$$

**Experiment (measured).** A 20×20 input, 3×3 filter, no padding, three strides:

| stride | formula $\lfloor(20-3)/s\rfloor+1$ | measured output |
|---|---|---|
| 1 | 18 | **18×18** ✓ |
| 2 | 9 | **9×9** ✓ |
| 3 | 6 | **6×6** ✓ |

![Stride](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_stride.png)

*The identical input and filter, only the step size changed. Larger strides produce smaller
outputs faster than padding or filter size alone would — this is one of the two standard
ways to downsample a feature map inside a CNN, the other being pooling
(see [pooling-operation.md](pooling-operation.md)).*

---

## 8. Convolving a multi-channel image

A color image has shape $H \times W \times 3$. A filter for it has a matching shape,
$f \times f \times 3$ — and critically, applying that filter still produces a single 2D
output, not three. The convolution sums over the channel dimension as well as the two
spatial dimensions:

$$Y_{i,j} = \sum_{u,v}\sum_{c=0}^{C_{in}-1} X_{i+u,\,j+v,\,c}\; K_{u,v,c}$$

**Experiment (measured).** A 16×16×3 image with three solid colored blocks (red,
green, blue), convolved with a single 3×3×3 filter whose weights are entirely on the red
channel:

- input shape: **(16, 16, 3)**
- kernel shape: **(3, 3, 3)**
- output shape: **(16, 16)** — a single map, one number per spatial position, exactly as the
  formula predicts.

![RGB convolution](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/08_rgb_convolution.png)

*(a) The three-block input. (b) A filter that only weights the red channel. (c) The output —
strongly positive exactly over the red block and near zero everywhere else, confirming the
filter is doing what its weights say: reading only from the channel it was given nonzero
weight on, and — this is the point of the section — collapsing all three input channels into
one output map per filter.*

---

## 9. Multiple filters: the output volume

A layer is not one filter; it is a **set** of filters, each independently convolved with the
full input volume and each producing its own 2D map. Stacking those maps along a new axis
gives the layer's output:

$$\text{input } (H, W, C_{in}) \ \ \xrightarrow{\ n \text{ filters, each } (f,f,C_{in})\ } \ \ \text{output } (H', W', n)$$

The output **depth equals the number of filters**, regardless of the input's depth. This is
how a network's channel count grows from 3 (RGB) in the input to 64, 128, 256... in deeper
layers — depth is a design choice made by picking how many filters each layer has.

**Experiment (measured).** The same 16×16×3 image, six filters (two edge-detector-like, four
random):

- kernels shape: **(6, 3, 3, 3)**
- output shape: **(16, 16, 6)** — depth exactly equals the filter count, 6.

![Multiple filters, one volume](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/09_multiple_filters_volume.png)

*(a) The output drawn as a literal 3D volume — six 16×16 slices stacked along a "filter
index" axis. (b) The same six slices laid flat, each one a genuinely different feature map
because each filter has different weights and therefore responds to different structure in
the input. Depth in a CNN is not a property of the image; it is a property of how many
filters you chose to learn.*

---

## 10. Filters are learned, not designed

Every kernel used above — Sobel, the red-channel filter, the random six — was hand-specified
to make a point. In an actual CNN, filters start as small random numbers and are updated
purely by gradient descent on the training loss, exactly like every other weight in the
network. No one tells the network to build an edge detector; if edge detection reduces the
loss, gradient descent finds it anyway.

**Backpropagation through convolution.** Given the gradient of the loss with respect to the
output feature map, $dL/dY$, the gradient with respect to the kernel is itself a convolution
— of the input with the output gradient — and the gradient with respect to the input is a
(zero-padded, kernel-flipped) convolution of the output gradient with the kernel:

$$\frac{\partial L}{\partial K_{u,v}} = \sum_{i,j} \frac{\partial L}{\partial Y_{i,j}}\, X_{i+u,j+v}, \qquad \frac{\partial L}{\partial X_{i+u,j+v}} \mathrel{+}= \frac{\partial L}{\partial Y_{i,j}}\, K_{u,v}$$

**Verification (measured), not assumed.** The backward pass above was implemented directly
and checked against finite-difference numerical gradients — the least trustable-until-proven
part of any from-scratch neural net code:

| | max |analytic − numerical| | relative to gradient magnitude |
|---|---|---|
| kernel gradient | $4.2\times10^{-11}$ | $1.2\times10^{-11}$ |
| input gradient | $6.0\times10^{-11}$ | $2.2\times10^{-11}$ |

![Gradient check](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/10_gradient_check.png)

*Every point is one gradient component, analytic value against a central-difference
numerical estimate ($\epsilon=10^{-5}$). Both panels sit exactly on the diagonal; the
residual is floating-point noise, not a bug.*

**A filter learned from scratch, measured.** Starting from random noise, minimizing the
squared error between a random kernel's output and a fixed target kernel's output — the
Sobel-X detector from [section 4](#4-convolution-as-edge-detection) — using nothing but the
gradient computed above and plain gradient descent:

- initial loss: **0.433**
- final loss (300 steps): **0.00085** — a **508×** reduction
- cosine similarity between the learned kernel and the true Sobel-X kernel: **0.928**

![A filter learned by backprop](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/11_filter_learned_by_backprop.png)

*(a) Training loss over 300 gradient steps, dropping over two and a half orders of
magnitude. (b) The kernel's actual weights at four points during training. Starting from
unstructured noise, the negative-left/positive-right pattern that defines a vertical-edge
detector visibly emerges by step 50 and sharpens by step 300 — nobody wrote "detect edges"
anywhere in this code; it fell out of minimizing squared error by gradient descent, which is
the entire mechanism by which a real CNN's filters end up detecting edges, corners, and
textures without ever being told to.*

---

## 11. From edges to corners: why depth matters

Section 1 promised that early layers detect primitives and deeper layers combine them into
more complex structure. This section builds the smallest possible working example of that
claim rather than asserting it.

**Experiment (measured).** Layer 1: convolve an image with both Sobel kernels and take the
magnitude of each response, giving a vertical-edge map and a horizontal-edge map. Layer 2: a
simple local-energy filter (a 5×5 box average) applied to each of those two maps, then
**multiplied together**. A position can only score high in the product if it has strong
local energy in *both* the vertical-edge map and the horizontal-edge map simultaneously —
which is precisely the definition of a corner. No layer here was told what a corner is.

- true corners on the test square: **4**
- corner-like blobs detected by layer 2 (threshold at half the peak response, connected-component count): **4** — found at all four correct locations.

![From edges to corners](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/12_feature_hierarchy.png)

*(a) The input square. (b), (c) Layer 1's two edge maps — each on its own only marks two of
the square's four sides (a vertical filter cannot see horizontal edges and vice versa).
(d) Layer 2, built from nothing but (b) and (c): four sharp, isolated peaks, exactly at the
square's four corners. Neither individual layer-1 map contains a corner detector; the corner
detector exists only in how the two combine. This is the entire logic behind stacking
convolutional layers — each additional layer's receptive field spans a larger region of the
original image and can represent combinations that no single earlier layer could.*

![Summary of measured results](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/13_summary_table.png)

*Every row above is a number produced by running the code in this document, not a claim
taken on faith.*

---

## 12. Implementation from scratch

A single-channel convolution, forward and backward, in plain NumPy:

```python
import numpy as np

def conv2d(img, kernel, stride=1, padding=0):
    """'Valid' cross-correlation — what deep learning calls 'convolution'.
       img: (H,W). kernel: (kh,kw). Returns (H',W')."""
    if padding > 0:
        img = np.pad(img, padding)
    H, W = img.shape
    kh, kw = kernel.shape
    oh = (H - kh) // stride + 1
    ow = (W - kw) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            r, c = i * stride, j * stride
            out[i, j] = np.sum(img[r:r + kh, c:c + kw] * kernel)
    return out


def conv2d_grad(img, kernel, dout, stride=1):
    """Gradient of a scalar loss w.r.t. img and kernel, given dL/d(out)."""
    H, W = img.shape
    kh, kw = kernel.shape
    oh, ow = dout.shape
    dimg = np.zeros_like(img)
    dker = np.zeros_like(kernel)
    for i in range(oh):
        for j in range(ow):
            r, c = i * stride, j * stride
            dker += dout[i, j] * img[r:r + kh, c:c + kw]
            dimg[r:r + kh, c:c + kw] += dout[i, j] * kernel
    return dimg, dker
```

Extending to multiple filters over multiple input channels — the general case from
[sections 8-9](#8-convolving-a-multi-channel-image) — is a matter of summing over the
channel axis and looping over filters:

```python
def conv2d_multi(img, kernels, stride=1, padding=0):
    """img: (H,W,Cin). kernels: (n_filters, kh, kw, Cin). Returns (H',W',n_filters)."""
    if padding > 0:
        img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)))
    H, W, Cin = img.shape
    n, kh, kw, _ = kernels.shape
    oh = (H - kh) // stride + 1
    ow = (W - kw) // stride + 1
    out = np.zeros((oh, ow, n))
    for f in range(n):
        for i in range(oh):
            for j in range(ow):
                r, c = i * stride, j * stride
                out[i, j, f] = np.sum(img[r:r + kh, c:c + kw, :] * kernels[f])
    return out
```

The framework equivalents — note that both take *filter count* and *kernel size* as their
primary arguments, exactly the two quantities that fixed the output shape in
[sections 5](#5-output-size-the-n--f--1-formula) and
[9](#9-multiple-filters-the-output-volume):

```python
# PyTorch
layer = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                        stride=1, padding=1)   # padding=1 gives "same" for a 3x3 filter

# Keras
layer = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
```

Both are drastically faster than the loops above in practice, because real implementations
lower convolution to a small number of large matrix multiplications (`im2col`) or use
direct convolution kernels on GPU hardware — but they compute exactly the sliding
dot-product described in [section 3](#3-the-convolution-operation-worked-by-hand).

---

## 13. Practical guidance

- **Default to 3×3 filters with "same" padding and stride 1** inside a network's main body;
  it is the overwhelming convention in modern architectures and, per
  [section 6](#6-padding), avoids the compounding spatial shrinkage of "valid" convolutions.
- **Use stride (or pooling) to downsample deliberately**, not as a side effect of skipping
  padding. A stride-2 convolution and a 2×2 max-pool are the two standard ways to halve
  spatial resolution; see [pooling-operation.md](pooling-operation.md).
- **Increase filter count (depth), not filter size, to add capacity** in deeper layers. Depth
  growth (64→128→256...) is cheap in parameters relative to growing kernel size, and stacked
  small filters (two 3×3 layers) cover the same receptive field as one larger filter (5×5)
  with fewer parameters and an extra nonlinearity in between.
- **A 1×1 convolution changes channel depth without touching spatial size** — useful for
  cheaply mixing information across channels or reducing depth before an expensive larger
  convolution (the "bottleneck" pattern).
- **Let backpropagation design the filters.** There is no need, and generally no benefit, to
  hand-initialize filters as edge detectors; [section 10](#10-filters-are-learned-not-designed)
  shows gradient descent finds them from random noise on its own, and networks initialized
  this way still learn edge- and orientation-selective filters in their first layer as a
  matter of course.

---

## 14. Key takeaways

1. Convolution replaces full connectivity with a small, shared, sliding filter — exploiting
   that image features are local and that the same feature can appear anywhere in the frame.
2. One output value is one dot product between the filter and a patch of the input, verified
   here to be an exact match between hand arithmetic and code: $-2$ both ways.
3. Output size follows $n-f+1$ (stride 1, no padding) or more generally
   $\lfloor(n+2p-f)/s\rfloor+1$. Checked against 123 (input, filter) pairs with zero
   deviation, and against three strides with zero deviation.
4. Convolution shrinks the input for $f>1$. Five stacked 3×3 "valid" layers left only 71.2%
   of the original area; "same" padding, adding $(f-1)/2$ border pixels, holds it at 100%.
5. A filter over a $C_{in}$-channel input has shape $(f,f,C_{in})$ and produces **one** 2D
   map per filter — channels are summed, not stacked. Measured: a 3×3×3 kernel on a
   16×16×3 image gives a 16×16 output, not 16×16×3.
6. A layer's output depth equals its number of filters, independent of input depth. Measured:
   six filters on a 16×16×3 image give a 16×16×6 output.
7. Backprop through convolution is itself expressible as convolutions. The hand-derived
   gradient matched numerical finite differences to within $6\times10^{-11}$ absolute error.
8. Filters are not designed — they are learned. Starting from random noise and minimizing
   squared error against a target output alone, gradient descent recovered a filter with
   92.8% cosine similarity to the true Sobel-X kernel and reduced the loss 508×, with no
   information about edges given anywhere in the training procedure.
9. Depth is what turns primitives into concepts. Two edge maps, neither containing a corner
   detector on its own, combined by a second layer into a detector that found all four of a
   test square's corners and nothing else.

---

## 15. Further reading

- Hubel, D. H., & Wiesel, T. N. (1962). *Receptive Fields, Binocular Interaction and
  Functional Architecture in the Cat's Visual Cortex.* The Journal of Physiology, 160(1),
  106–154. — The biological basis for local receptive fields and hierarchical feature
  extraction referenced in [section 1](#1-why-a-specialized-operation-for-images).
- Fukushima, K. (1980). *Neocognitron: A Self-organizing Neural Network Model for a Mechanism
  of Pattern Recognition Unaffected by Shift in Position.* Biological Cybernetics, 36(4),
  193–202. — An early computational model directly inspired by Hubel & Wiesel.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-Based Learning Applied
  to Document Recognition.* Proceedings of the IEEE, 86(11), 2278–2324. — LeNet-5 and the
  modern formulation of the convolutional layer trained end-to-end by backpropagation.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet Classification with Deep
  Convolutional Neural Networks.* NeurIPS. — AlexNet, the paper that made CNNs the default
  approach for image tasks.
- Dumoulin, V., & Visin, F. (2016). *A guide to convolution arithmetic for deep learning.*
  arXiv:1603.07285. — A thorough reference for the output-size, padding, and stride formulas
  used in [sections 5-7](#5-output-size-the-n--f--1-formula).
- Zeiler, M. D., & Fergus, R. (2014). *Visualizing and Understanding Convolutional Networks.*
  ECCV. — Direct visual evidence for the edge → texture → part → object hierarchy invoked in
  [section 1](#1-why-a-specialized-operation-for-images) and
  [section 11](#11-from-edges-to-corners-why-depth-matters).
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, chapter 9. MIT Press.
  — Textbook treatment of convolution, including the cross-correlation-vs-convolution
  distinction noted in [section 3](#3-the-convolution-operation-worked-by-hand).

---

*Part of a series of deep-learning study notes. Related:
[gradient-descent.md](gradient-descent.md) ·
[momentum-optimization.md](momentum-optimization.md) ·
[adam.md](adam.md) ·
[pooling-operation.md](pooling-operation.md) ·
[cnn-architectures.md](cnn-architectures.md)*
