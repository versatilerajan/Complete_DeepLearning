# Designing Better Architectures for Grid-Structured Data

*Companion notes expanding on MIT 6.7960 (Deep Learning) — the lecture on encoding structural assumptions about data into model architecture, with a focus on images, video, and continuous signals.*

---

## Table of Contents

1. [Why Build Better Architectures?](#1-why-build-better-architectures)
2. [Convolutional Neural Networks: The Right Tool for Grids](#2-convolutional-neural-networks-the-right-tool-for-grids)
3. [Core Building Block #1: Convolutional Kernels](#3-core-building-block-1-convolutional-kernels)
4. [Core Building Block #2: Pooling](#4-core-building-block-2-pooling)
5. [Pyramids: Reasoning at Multiple Scales](#5-pyramids-reasoning-at-multiple-scales)
6. [The Architecture Zoo](#6-the-architecture-zoo)
   - [Skip Connections & ResNet](#skip-connections--resnet)
   - [Encoder-Decoder & U-Net](#encoder-decoder--u-net)
   - [3D Convolutions for Video](#3d-convolutions-for-video)
7. [Neural Fields and Positional Encoding](#7-neural-fields-and-positional-encoding)
8. [Putting It All Together](#8-putting-it-all-together)
9. [Further Reading](#9-further-reading)

---

## 1. Why Build Better Architectures?

Every model architecture makes a bet about the *structure* of the data it will see. A generic fully-connected network makes almost no bet at all — it treats every input dimension as independent and unrelated, and has to *learn* any structure (locality, symmetry, periodicity) entirely from data. A well-designed architecture instead **bakes a hypothesis about the data directly into the model**. This hypothesis is called an **inductive bias**.

Why does this matter so much in practice?

- **Data efficiency.** If the model already "knows" that nearby pixels are related, it doesn't have to rediscover that fact from thousands of examples — it can spend its learning capacity on the actual task.
- **Better generalization.** A model that has to memorize a pattern independently at every possible location or orientation will generalize poorly to positions it hasn't seen. A model that structurally *guarantees* the pattern is recognized everywhere generalizes far better.
- **Fewer parameters, cheaper training.** Structural constraints (like reusing the same weights everywhere) collapse a huge number of free parameters into a much smaller, better-constrained set.

A clean example from the lecture is the **SIREN** model, which uses sinusoidal activation functions instead of ReLU specifically because natural signals (images, audio) contain smooth, periodic structure — this is a *strong prior* that lets the network fit fine detail with far less data than a generic architecture would need. The general lesson: **the closer your architecture's assumptions match the true structure of the data, the less data and compute you need to learn a good model.**

![Why architecture matters: fully-connected vs convolutional connectivity](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/inductive_bias.png)

*Left: a fully-connected layer assumes nothing about spatial structure — every output depends on every input, and the model must learn locality from scratch. Right: a convolutional layer bakes in two assumptions directly — locality (only nearby pixels matter for a given output) and translation invariance (the same pattern-detector is reused everywhere).*

---

## 2. Convolutional Neural Networks: The Right Tool for Grids

Images (and similarly, audio spectrograms, video, and other grid-like data) have two properties that a good architecture should exploit:

1. **Local structure matters.** A pixel's meaning is best understood in the context of its neighbors — an edge, a corner, a texture — not in relation to a pixel on the opposite side of the image.
2. **Patterns can appear anywhere.** A cat's ear looks like a cat's ear whether it's in the top-left or bottom-right of the photo. The *detector* for "ear-like edges" shouldn't have to be learned separately for every possible location.

CNNs encode both assumptions directly:

- **Local receptive fields** — each unit in a convolutional layer only looks at a small patch of the input (its receptive field), not the whole image.
- **Weight sharing** — the *same* small filter (kernel) is applied at every spatial location. This is what produces **translation equivariance**: if you shift the input, the output feature map shifts by exactly the same amount.

![Translation equivariance demonstration](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/translation_equivariance.png)

*A synthetic square shape is detected by the same edge-detecting kernel regardless of where it sits in the image — shifting the input shifts the resulting feature map identically. This property is what lets a CNN recognize an object anywhere in the frame after seeing it in only a few positions during training.*

Stacking many convolutional layers lets a network build up from simple local patterns (edges, corners) in early layers to increasingly complex, larger-scale concepts (textures, parts, objects) in deeper layers — each layer's receptive field covers more of the original image than the last.

---

## 3. Core Building Block #1: Convolutional Kernels

A **kernel** (or filter) is a small grid of learnable numbers — commonly 3×3 or 5×5 — that slides across the input, computing a dot product with the patch of input it currently covers.

![The convolution operation, step by step](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/convolution_operation.png)

*At each position, the kernel is overlaid on a patch of the input, an element-wise product is taken, and the results are summed into a single output value. Sliding this window across the whole input produces the output feature map. In deep learning frameworks this operation is technically cross-correlation (no kernel flipping), though it's universally called "convolution."*

Early hand-designed kernels (like the classic Sobel operator) were built by hand to detect specific patterns such as edges:

![Edge detection using a real kernel](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/edge_detection_kernel.png)

*A Sobel-style kernel highlights vertical intensity changes — you can see it lighting up the left and right edges of the shapes. In a trained CNN, the network doesn't need this kernel hand-designed: gradient descent discovers useful filters like this (and many more complex ones) automatically, purely from data.*

A single convolutional layer typically learns **many** kernels in parallel (e.g., 64 different 3×3 filters), each producing its own feature map — one might fire on vertical edges, another on horizontal edges, another on a specific color transition, and so on. Stacking these across layers is how the network builds a rich hierarchy of learned features.

---

## 4. Core Building Block #2: Pooling

After convolution, **pooling** layers reduce the spatial resolution of the feature maps — most commonly via **max pooling**, which keeps only the strongest activation in each small window.

![Max pooling operation](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/pooling_operation.png)

*A 2×2 max-pooling window slides across the input with a stride of 2, keeping only the largest value from each window. This halves both spatial dimensions.*

Why pool at all?

- **Computational efficiency.** Fewer spatial locations means less computation in every subsequent layer.
- **Spatial stability (a limited translation invariance).** If the input shifts by a pixel or two, the max value in each pooling window is often unchanged — the network becomes less sensitive to *exactly* where a feature appears, which is usually desirable (a cat is still a cat if the photo is nudged slightly).
- **Growing the receptive field.** After pooling, the *next* convolutional layer's fixed-size kernel effectively "sees" a larger region of the original image, since each pixel in the pooled map already summarizes a neighborhood.

The trade-off: pooling throws away precise spatial information. This becomes important later — see [Skip Connections](#skip-connections--resnet) below, which exist partly to compensate for information pooling discards.

---

## 5. Pyramids: Reasoning at Multiple Scales

Objects in images appear at wildly different scales — a face might fill the whole frame in a portrait or occupy just a few pixels in a crowd photo. A single fixed-resolution view of the image can't handle both cases well. **Pyramids** address this by representing the same content at a sequence of progressively coarser resolutions.

![Image / feature pyramid across multiple scales](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/image_pyramid.png)

*Each level blurs and downsamples the previous one. Fine levels retain detail useful for precise localization; coarse levels retain global structure and long-range context cheaply (since there are far fewer pixels to process).*

CNNs naturally build something like a pyramid internally: every time you apply a conv + pooling stage, the spatial resolution shrinks while the receptive field (the region of original input a single output "sees") grows. Deeper layers therefore automatically encode more global, coarse information, while early layers retain fine local detail — this observation underlies architectures like **Feature Pyramid Networks (FPN)**, which explicitly combine feature maps from multiple depths so a detector can recognize both small and large objects well.

---

## 6. The Architecture Zoo

With the basic building blocks (conv, pooling) established, real-world architectures combine them in different topologies to solve different problems. A few of the most influential patterns:

### Skip Connections & ResNet

As networks get deeper, training gets **harder**, not easier — gradients can vanish or explode across many layers, and even in the best case, a deeper network can struggle to learn to simply "do nothing extra" on top of a shallower solution that already works. **Residual connections** (introduced in ResNet) fix this with a strikingly simple trick: instead of asking a block of layers to learn the full output directly, ask it to learn only the **residual** — the *difference* from its input — and add the original input back in.

![Residual connections in ResNet, and skip connections in U-Net](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/skip_connections.png)

*Left (ResNet): the block computes F(x), and the output is F(x) + x. If the ideal transformation really is close to the identity, the network can drive F(x) toward zero — a much easier optimization target than learning the identity mapping from scratch through several nonlinear layers. This simple change is a major reason very deep networks (50, 100+ layers) became trainable at all.*

### Encoder-Decoder & U-Net

Many tasks (image segmentation, denoising, translation between image domains) require the output to be the **same spatial size** as the input, but the network still benefits from the compression-then-expansion pattern: an **encoder** progressively downsamples the input into a compact representation, and a **decoder** progressively upsamples it back out.

The problem: aggressive downsampling in the encoder throws away exactly the fine-grained spatial detail (precise edges, boundaries) needed to reconstruct a crisp, accurate output. **U-Net** solves this by adding skip connections directly from each encoder stage to the matching-resolution decoder stage, letting fine detail bypass the bottleneck entirely.

*(See the right panel of the image above.)* Each horizontal green connection copies the encoder's feature map at a given resolution directly across to the decoder stage that's about to upsample back to that same resolution, where it's concatenated with the upsampled features. The result: the decoder gets both the high-level, globally-informed representation from the bottleneck **and** the precise local detail preserved by the skip connections — the best of both worlds.

### 3D Convolutions for Video

Video is naturally a 3D grid: height, width, and **time**. A 3D convolution simply extends the same idea one dimension further — the kernel is now a small cuboid that slides across height, width, *and* time simultaneously, letting the network learn spatio-temporal patterns (like motion) directly, the same way a 2D kernel learns spatial patterns.

![Generic encoder-decoder and 3D convolution for video](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/encoder_decoder_3dconv.png)

*Left: the generic encoder-decoder pattern — compress to a bottleneck, then reconstruct. Right: a 3D convolution kernel (orange) slides through a video volume (blue) along all three axes at once, so a single filter can learn to detect, e.g., "an edge moving rightward" — a pattern that only exists across consecutive frames, not within any single frame.*

---

## 7. Neural Fields and Positional Encoding

So far, every architecture above assumes the input is a discrete grid — a fixed array of pixels or voxels. **Neural fields** take a fundamentally different approach: instead of *storing* a signal as a grid of samples, represent it as a **continuous function**, and let a neural network learn that function directly.

The most well-known example is **NeRF (Neural Radiance Fields)**: rather than storing a 3D scene as a voxel grid or mesh, NeRF trains a small MLP to map *any* continuous 3D coordinate (plus a viewing direction) directly to a color and density at that point.

![Neural fields: coordinates to MLP to signal, and positional encoding](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/neural_fields.png)

*Left: the network itself becomes the representation of the scene — query any coordinate, even between the "pixels" of a traditional grid, and get back a value. There's no fixed resolution; the same trained network can be rendered at any zoom level. Right: plain coordinates change too smoothly for an MLP with a limited number of layers to represent sharp, high-frequency detail (fine textures, crisp edges). Positional encoding maps each coordinate through a bank of sine and cosine functions at increasing frequencies before feeding it to the network, giving the MLP the high-frequency "building blocks" it needs to represent fine detail — without this trick, NeRF-style networks produce noticeably blurry results.*

This connects directly back to [Section 1](#1-why-build-better-architectures): positional encoding is itself an inductive bias — a structural assumption (that natural signals contain a spread of frequency content) baked directly into how the input is prepared for the network, dramatically improving what the same-sized MLP is able to represent.

---

## 8. Putting It All Together

| Concept | What it encodes | Where it shows up |
|---|---|---|
| Convolution (weight sharing) | Locality + translation equivariance | Every CNN layer |
| Pooling | "What" matters more than "exactly where" | Classic CNN stacks (LeNet, VGG, ResNet) |
| Pyramids | Multi-scale reasoning | Feature Pyramid Networks, image processing pipelines |
| Skip / residual connections | Let gradients and fine detail bypass deep stacks | ResNet, U-Net, most modern deep architectures |
| Encoder-decoder | Compress to essentials, then reconstruct | Autoencoders, U-Net, segmentation & generation models |
| 3D convolution | Locality + equivariance across space **and** time | Video classification, action recognition |
| Neural fields + positional encoding | Continuous, resolution-free representation of a signal | NeRF, signal/shape representation, SIREN |

The throughline across all of these: **good architectures aren't found by making the network bigger and hoping — they come from correctly identifying the structure in your data (grid, sequence, continuous coordinate, temporal) and choosing an architecture whose assumptions match that structure.** This is exactly the same design principle whether you're placing a 3×3 kernel over an image or choosing to feed coordinates through a Fourier-feature encoding before an MLP.

---

## 9. Further Reading

- LeCun, Y. et al. (1998). *Gradient-Based Learning Applied to Document Recognition* — the original CNN / LeNet paper.
- He, K. et al. (2015). *Deep Residual Learning for Image Recognition* — ResNet and residual connections.
- Ronneberger, O. et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.*
- Mildenhall, B. et al. (2020). *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.*
- Sitzmann, V. et al. (2020). *Implicit Neural Representations with Periodic Activation Functions* — the SIREN paper.
- Tran, D. et al. (2015). *Learning Spatiotemporal Features with 3D Convolutional Networks.*
- Lin, T.-Y. et al. (2017). *Feature Pyramid Networks for Object Detection.*
- MIT 6.7960 (Deep Learning) course materials, MIT OpenCourseWare / course site — the lecture this document expands on.

---

*Diagrams in this document were generated programmatically to illustrate the concepts discussed above, and are hosted in this repo's [`images/`](https://github.com/versatilerajan/deepcontent/tree/main/images) folder.*
