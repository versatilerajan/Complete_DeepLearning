# ANN vs CNN: Same Primitive, Different Wiring

[cnn-intro.md](cnn-intro.md) and [padding-and-strides.md](padding-and-strides.md) covered why plain ANNs struggle with images and how a convolution's sliding window actually operates. This document steps back and asks a sharper question: if a CNN's filter and an ANN's neuron are, underneath, doing the exact same arithmetic — a dot product, a bias, an activation — then what specifically makes a CNN's parameter count stay flat as images grow, while an ANN's explodes? The answer turns out to have two separable parts, and this document isolates them with real, instantiated models rather than formulas alone. One CNN filter applied at one image location is verified here to produce numerically identical output to an ANN neuron applied to the same flattened patch — to float64 precision, across 720 separate position-and-filter checks. Local connectivity alone is *not* what saves parameters: a "locally connected" layer with the same small receptive field as a CNN, but without weight sharing, is measured here scaling its parameter count right alongside a fully dense layer — 169,280 parameters at a 48×48 image, versus the CNN's constant 80. Weight *sharing* specifically is the mechanism, confirmed by building all three layer types as real, gradient-checked models and counting their actual parameter arrays. This document is also the direct setup for the next one in the series, on backpropagation through a CNN.

> **Series note.** This sits alongside [cnn-intro.md](cnn-intro.md) and [padding-and-strides.md](padding-and-strides.md) as background before [cnn-backprop.md](cnn-backprop.md), which this document exists specifically to set up.

---

## Table of Contents

1. [Recap: why plain ANNs struggle with images](#1-recap-why-plain-anns-struggle-with-images)
2. [The shared computational primitive](#2-the-shared-computational-primitive)
3. [Experiment: verifying the primitive is actually identical](#3-experiment-verifying-the-primitive-is-actually-identical)
4. [What actually keeps CNN parameter count constant](#4-what-actually-keeps-cnn-parameter-count-constant)
5. [Experiment: local connectivity alone is not enough](#5-experiment-local-connectivity-alone-is-not-enough)
6. [Experiment: does the parameter difference show up in practice?](#6-experiment-does-the-parameter-difference-show-up-in-practice)
7. [Framework usage](#7-framework-usage)
8. [Key takeaways](#8-key-takeaways)
9. [Further reading](#9-further-reading)

---

## 1. Recap: why plain ANNs struggle with images

[cnn-intro.md](cnn-intro.md) covered three concrete problems with feeding images into a plain ANN: high computational cost from a parameter count that scales with image resolution, an increased risk of overfitting that comes directly from having that many parameters relative to the size of a typical training set, and the loss of spatial information caused by flattening a 2-D grid into a 1-D vector before the network ever sees it. This document does not re-run those experiments — see [cnn-intro.md](cnn-intro.md) Sections 2 and 4 for the measured parameter explosion (up to 112,348× at 512×512 resolution) and the measured effect of scrambling pixel order. What follows here is a different, complementary question: given that a CNN and an ANN are built from what the video calls the same underlying principles, what exactly is the structural change that fixes these three problems?

## 2. The shared computational primitive

Strip away the sliding-window framing, and a single CNN filter evaluated at a single position computes:

$$y_{i,j} = \sigma(\mathbf{w} \cdot \mathbf{x}_{i,j} + b)$$

where $\mathbf{x}_{i,j}$ is the flattened patch of the input under the kernel at position $(i,j)$, $\mathbf{w}$ is the flattened kernel, $b$ is a bias, and $\sigma$ is an activation function such as ReLU. This is line-for-line the same formula as a single ANN neuron:

$$y = \sigma(\mathbf{w} \cdot \mathbf{x} + b)$$

![The shared primitive](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_shared_primitive.png)

*Side by side, the only notational difference is a subscript: the ANN neuron's $\mathbf{x}$ is the entire flattened input, applied once; the CNN filter's $\mathbf{x}_{i,j}$ is a small local patch, but the same weight vector $\mathbf{w}$ is reused at every $(i,j)$ position the filter visits. The dot-product-then-bias-then-activation computation itself does not change at all.*

This is worth taking seriously rather than treating as a loose analogy — Section 3 checks it directly rather than asserting it.

## 3. Experiment: verifying the primitive is actually identical

Take a real 8×8 image, a real 3×3 kernel, and a bias. Compute the response at one interior position two different ways: (1) the way a CNN would — multiply the patch and kernel element-wise and sum, exactly the sliding-window operation from [cnn-intro.md](cnn-intro.md) Section 3 — and (2) the way an ANN neuron would — flatten both the patch and the kernel into vectors and take their dot product. Both get a bias added and a ReLU applied identically.

| Method | Output |
|---|---|
| CNN-style (element-wise multiply, sum) | 1.3453179811542804 |
| ANN-style (flatten, dot product) | 1.3453179811542806 |
| Matrix-multiply style (`x @ w`) | 1.3453179811542806 |

The three outputs agree to $2.2\times10^{-16}$ — the float64 precision floor, arising purely from the two methods summing the same nine products in a different order (`np.sum` over a 2-D array versus a 1-D dot product), not from any conceptual difference. Repeating this check across 20 random filters and all 36 positions a 3×3 kernel visits on this image (720 total position-and-filter combinations):

| | Result |
|---|---|
| Position/filter combinations checked | 720 |
| Mismatches beyond float64 precision | **0** |

Every single check matches. A CNN filter is not "inspired by" or "analogous to" an ANN neuron — applied at one fixed position, it computes the identical function.

## 4. What actually keeps CNN parameter count constant

If the underlying computation is identical, the entire difference between an ANN layer and a CNN layer must be in the *wiring* — which inputs feed which outputs, and whether different outputs are allowed to share the same weights. There are, in fact, three distinct wiring choices worth separating, not two:

![Three ways to wire a layer](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_three_way_wiring.png)

*Dense (left): every output unit connects to every input pixel, with its own independent weight for each connection. Locally connected (middle): each output only looks at a small local patch of the input — exactly the same receptive field a CNN filter would have — but each output position still gets its own, independently-learned set of weights; nothing is shared between positions. Weight-shared CNN (right): the same small receptive field as the middle case, but now the identical weight vector is reused at every position.*

It is tempting to assume that restricting connectivity to a small local patch (the middle case) is what saves parameters, since that is the more visually obvious change from the dense case. Section 5 tests this directly, and it turns out to be the wrong explanation.

## 5. Experiment: local connectivity alone is not enough

Three real layer types were implemented and gradient-checked (backward pass verified against numerical differentiation, maximum relative error under $1.1\times10^{-8}$ for the locally-connected layer): a dense layer, a locally-connected layer (same 3×3 receptive field as a CNN, independent weights per output position), and a standard weight-shared convolutional layer. Their first-layer parameter counts were then measured — not estimated — by instantiating each as a real object and counting its actual weight array sizes, at five input sizes from 8×8 to 48×48, and cross-checked against the closed-form parameter-count formula for each (zero discrepancies at every size tested).

![Three-tier parameter comparison](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_three_tier_params.png)

*Log scale. The dense layer's parameter count grows as expected. The locally-connected layer's parameter count grows at almost exactly the same rate — despite looking at only a 3×3 patch of the input for each output, exactly like the CNN — because a new independent set of weights is required at every one of the (growing number of) output positions. Only the weight-shared CNN stays flat, at a constant 80 parameters regardless of whether the input is 8×8 or 48×48.*

| Input size | Dense first layer | Locally connected first layer | Weight-shared CNN first layer |
|---|---|---|---|
| 8×8 | 4,160 | 2,880 | 80 |
| 16×16 | 16,448 | 15,680 | 80 |
| 24×24 | 36,928 | 38,720 | 80 |
| 32×32 | 65,600 | 72,000 | 80 |
| 48×48 | 147,520 | **169,280** | **80** |

At the largest size tested, the locally-connected layer needs **2,116× more** parameters than the weight-shared CNN, despite having an identical receptive field at every output position — the only difference between the two is whether that receptive field's weights are reused or relearned from scratch at each position. Local connectivity by itself buys almost nothing here; **weight sharing is doing essentially all of the work** in keeping the CNN's parameter count independent of input size, which is precisely the claim the video makes in its 12:00–15:48 demonstration, now isolated from the (much smaller, and separately real) contribution of restricting connectivity to a local patch.

## 6. Experiment: does the parameter difference show up in practice?

Parameter count differences of this size raise an obvious follow-up: does it actually matter for a model trained on real data, in the way the video's "potential for overfitting" claim suggests? All three architectures were trained on the same task — scikit-learn's 8×8 digit dataset, 5 seeds each — to check.

![Overfitting comparison](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_overfitting_comparison.png)

*Left: total parameter count for each full architecture (first layer plus the same pooling-and-classifier structure) on this specific 8×8 task. Right: the gap between training and test accuracy, averaged over 5 seeds.*

| Architecture | Total params | Test accuracy | Train − test gap |
|---|---|---|---|
| MLP (dense) | 1,810 | 96.8% | 0.027 |
| Locally connected | 3,610 | 97.6% | 0.019 |
| Weight-shared CNN | **810** | **97.7%** | 0.023 |

This result needs to be reported honestly rather than forced into the cleanest possible story: the overfitting gaps here are all small (0.019–0.027) and do **not** cleanly track parameter count — the locally-connected model, despite having the most parameters of the three, showed the *smallest* train/test gap, not the largest. On this particular small, easy, 10-class task, none of the three architectures overfits dramatically. What the numbers do support clearly is a different, still meaningful claim: the weight-shared CNN reaches accuracy at least as good as either alternative (97.7%, edging out both) while using **less than half the parameters** of the dense model and roughly a **quarter** of the locally-connected model's. On this task, weight sharing's clearly demonstrated benefit is parameter efficiency without any loss of accuracy — not a dramatic reduction in overfitting, which would likely require a harder task or a smaller training set to show clearly (a natural extension for a future notes file, not claimed here).

## 7. Framework usage

Both Keras and PyTorch expose a locally-connected layer distinct from a normal convolution, which is worth knowing exists precisely because Section 5 shows it is not the same thing as a CNN layer, despite sharing the same local receptive field:

**Keras:**

```python
from tensorflow.keras.layers import Conv2D, LocallyConnected2D

# Standard weight-shared convolution -- constant parameter count
Conv2D(filters=8, kernel_size=(3, 3), activation='relu')

# Locally connected: same receptive field, but independent weights
# per output position -- parameter count scales with input size,
# exactly as measured in Section 5
LocallyConnected2D(filters=8, kernel_size=(3, 3), activation='relu')
```

**PyTorch** has no built-in locally-connected layer in its core `nn` module (it is convolution-first); a locally-connected layer is typically implemented by hand or via `nn.Unfold` to extract patches, followed by a per-position linear layer.

Neither snippet was executed — neither framework is installed in the environment used for this document — but the parameter-count behaviour they implement matches exactly what was independently built and verified in Section 5's from-scratch implementation.

## 8. Key takeaways

![Summary of all measurements](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_summary_table.png)

1. **A CNN filter and an ANN neuron compute the identical function.** Verified to float64 precision across 720 position-and-filter checks: dot product, add bias, apply activation — the arithmetic does not change between the two.
2. **The wiring, not the arithmetic, is what differs.** Three distinct wiring patterns exist — dense, locally connected, and weight-shared — and they are not the same two-way choice ("global" vs. "local") that they are sometimes informally reduced to.
3. **Local connectivity alone does not keep parameter count flat.** A locally-connected layer, with the exact same 3×3 receptive field as a CNN, still scales its parameter count with image size almost identically to a fully dense layer — measured here at 2,116× more parameters than the equivalent CNN layer at 48×48.
4. **Weight sharing is specifically what makes CNN parameter count independent of input size.** Measured directly: a constant 80 parameters in the first layer regardless of whether the input is 8×8 or 48×48, verified against both a closed-form formula and a real instantiated model's weight arrays.
5. **The measured effect on overfitting is real but should not be overstated.** On an easy digit-classification task, all three architectures showed similarly small train/test gaps; the clean, well-supported benefit of weight sharing here is parameter efficiency at matched or better accuracy, not a dramatic reduction in overfitting on this particular task.
6. **This sets up backpropagation through a CNN directly.** Since a filter application is the same primitive as a neuron, the gradient computation for a single position is identical to the ANN case already familiar from [gradient-descent.md](gradient-descent.md); what changes for a full CNN is that gradients from every position sharing a kernel must be accumulated onto that one shared set of weights — exactly the subject of [cnn-backprop.md](cnn-backprop.md).

## 9. Further reading

- **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*, Chapter 9 ("Convolutional Networks"). MIT Press. — Section 9.2 in particular formalizes the "three ideas" (sparse interactions, parameter sharing, equivariant representations) that structure the comparison in this document.
- **LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).** *Gradient-based learning applied to document recognition.* Proceedings of the IEEE, 86(11), 2278–2324. — Discusses locally-connected ("unshared") layers explicitly as an intermediate design point between full connectivity and shared convolution, in the context of LeNet-5's own architecture choices.
- **Chollet, F.** *Keras documentation: LocallyConnected2D layer* (keras.io). — The framework-level implementation referenced in Section 7.
- **Bishop, C. M., & Bishop, H. (2024).** *Deep Learning: Foundations and Concepts*, Chapter 10 ("Convolutional Networks"). Springer. — A modern treatment covering the parameter-sharing argument alongside the equivalent-computation framing used in Sections 2–3 of this document.

---

*Part of an ongoing deep learning notes series. Previous: [padding-and-strides.md](padding-and-strides.md). Next: [cnn-backprop.md](cnn-backprop.md).*
