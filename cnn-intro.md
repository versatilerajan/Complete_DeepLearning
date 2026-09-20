# Introduction to Convolutional Neural Networks (CNNs)

Every neural network covered earlier in this series — including the optimizers in [gradient-descent.md](gradient-descent.md), [momentum.md](momentum.md), [nesterov-accelerated-gradient.md](nesterov-accelerated-gradient.md), and [adagrad.md](adagrad.md) — was trained on a plain, fully-connected architecture (an ANN): every input feature connects to every unit in the next layer. That works fine for tabular data, but it breaks down badly for images, and this document explains exactly why and what CNNs do instead. A dense first layer's parameter count is measured here to explode from 100,480 at a small 28×28 image to over 100 million at a modest 512×512 image — while a convolutional layer's parameter count stays fixed at 896, regardless of image size. A from-scratch CNN and a matched plain network are then trained on the identical dataset with the identical pixels scrambled by one fixed permutation: the plain network's accuracy barely moves, while the CNN's drops measurably more — direct, computed evidence that the convolutional network is doing something the flattening-based network is not: actually using where each pixel sits relative to its neighbours. The second half of the document traces where this idea came from biologically — Hubel and Wiesel's 1959–1962 recordings from a cat's visual cortex — through Fukushima's Neocognitron (1980) to LeCun's LeNet-5 (1998) and the 2012 AlexNet result that triggered the deep learning boom.

> **Series note.** This is the first notes file in the series to leave optimizers and address architecture. It assumes the reader already has [gradient-descent.md](gradient-descent.md)'s background on how a network is trained; it does not assume any prior CNN-specific vocabulary. Later files in this series ([lenet.md](lenet.md), [alexnet.md](alexnet.md), and a practical [cat-vs-dog-classifier.md](cat-vs-dog-classifier.md) project) build directly on the concepts introduced here.

---

## Table of Contents

1. [Why not just use a plain ANN on images?](#1-why-not-just-use-a-plain-ann-on-images)
2. [Experiment: the parameter explosion, measured](#2-experiment-the-parameter-explosion-measured)
3. [What a convolution actually computes](#3-what-a-convolution-actually-computes)
4. [Experiment: does spatial arrangement actually matter?](#4-experiment-does-spatial-arrangement-actually-matter)
5. [Experiment: from edges to a trained filter bank](#5-experiment-from-edges-to-a-trained-filter-bank)
6. [The biological inspiration: the visual cortex](#6-the-biological-inspiration-the-visual-cortex)
7. [The Hubel and Wiesel cat experiments](#7-the-hubel-and-wiesel-cat-experiments)
8. [Simple cells, complex cells, and why CNNs are shaped the way they are](#8-simple-cells-complex-cells-and-why-cnns-are-shaped-the-way-they-are)
9. [From biology to a computer: Neocognitron, LeNet, AlexNet](#9-from-biology-to-a-computer-neocognitron-lenet-alexnet)
10. [Receptive fields: how depth lets a network see more](#10-receptive-fields-how-depth-lets-a-network-see-more)
11. [Applications](#11-applications)
12. [Key takeaways](#12-key-takeaways)
13. [Further reading](#13-further-reading)

---

## 1. Why not just use a plain ANN on images?

A plain artificial neural network (ANN) takes a fixed-length vector as input. An image is a 2-D (or 3-D, with colour channels) grid of pixels, so using an ANN on it requires *flattening*: reshaping the grid into one long vector before it ever reaches the first layer. This single step causes three separate problems, all real and all measurable, not just theoretical concerns:

**High computational cost.** A fully-connected layer's parameter count is (number of inputs) × (number of units in that layer). For an image, "number of inputs" is height × width × channels — a number that grows quadratically with resolution. Section 2 measures exactly how fast this grows.

**Overfitting risk.** More parameters relative to the amount of training data means more capacity to memorize the training set rather than learn generalizable patterns — a well-established relationship in machine learning generally, and a direct consequence of the parameter counts in Section 2.

**Loss of spatial arrangement.** Flattening imposes an arbitrary 1-D order on what was a 2-D grid. Pixel $(i, j)$ and its neighbour $(i, j+1)$ — physically adjacent in the image — may end up far apart in the flattened vector, and a plain ANN has no way to know they were ever adjacent to begin with: it treats the flattened vector exactly the same regardless of what order its entries came in. Section 4 tests this directly rather than just asserting it.

## 2. Experiment: the parameter explosion, measured

To quantify the first problem concretely: compare the parameter count of a single dense layer (mapping a flattened image to 128 hidden units) against a single convolutional layer (32 filters, each 3×3), at five realistic image sizes.

![Parameter explosion](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_param_explosion.png)

*Real arithmetic, not an illustration: parameters = (height × width × channels) × 128 + 128 for the dense layer, versus (3×3×channels)×32 + 32 for the conv layer — a fixed number that does not depend on height or width at all. Note the log scale: the dense layer's bar grows by roughly one order of magnitude every time the image doubles in linear size, while the conv layer's bar is flat.*

| Image size | Dense-layer parameters | Conv-layer parameters | Ratio |
|---|---|---|---|
| 28×28×1 | 100,480 | 320 | 314× |
| 64×64×3 | 1,572,992 | 896 | 1,756× |
| 128×128×3 | 6,291,584 | 896 | 7,022× |
| 224×224×3 (ImageNet standard) | 19,267,712 | 896 | 21,504× |
| 512×512×3 | 100,663,424 | 896 | **112,348×** |

At 512×512 resolution, the dense layer alone needs over **100 million** parameters just to produce its first 128 hidden units — more than many complete, deep, competitive CNN architectures use in total. The conv layer needs 896, and that number would still be 896 at 4096×4096. The mechanism behind this gap is shown next.

## 3. What a convolution actually computes

![Convolution mechanics](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_convolution_mechanics.png)

*A single step of 2-D convolution, computed exactly as shown: the 3×3 kernel is placed over the top-left 3×3 patch of the input, each pair of overlapping values is multiplied, and the nine products are summed to produce one output number. Sliding this same kernel to every valid position in the 5×5 input produces the full 3×3 output grid on the right. In an actual CNN the numbers inside the kernel are not fixed like this — they start random and are learned by gradient descent, exactly like every weight in the ANNs covered earlier in this series.*

The formula for one output position, with kernel $K$ of size $k \times k$ applied to input $X$ at position $(i, j)$:

$$\text{out}(i, j) = \sum_{u=0}^{k-1}\sum_{v=0}^{k-1} X(i+u,\, j+v)\cdot K(u, v)$$

The critical structural difference from a dense layer is visible directly in this formula: $K$ does not depend on $(i, j)$. The *same* $k \times k$ set of numbers is reused at every position in the image.

![ANN vs CNN wiring](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_ann_vs_cnn_wiring.png)

*Left: a dense layer connects every pixel to every hidden unit with its own independent weight — hence the parameter count scaling with image size. Right: a convolutional layer reuses one small kernel at every position, so the parameter count is fixed by the kernel size and the number of filters, never by the image's height or width. This single design choice — weight sharing across spatial position — is the entire reason for the gap measured in Section 2.*

## 4. Experiment: does spatial arrangement actually matter?

Section 1 claimed that flattening loses spatial arrangement and that this specifically hurts convolutional processing. This is directly testable: train a from-scratch CNN (one conv layer, ReLU, max-pool, then a fully-connected output) and a parameter-matched plain MLP on the same digit-classification dataset (scikit-learn's 8×8 handwritten digits) under two conditions — the images as-is, and every image scrambled by **one fixed random pixel permutation**, applied identically to every training and test image. If the claim is right, the MLP (which never treats its input as anything but an unordered vector) should barely notice; the CNN (whose convolution operation is only meaningful if neighbouring entries are actually spatially adjacent) should be measurably more affected.

![What the permutation does to an image](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_permutation_example.png)

*The exact transformation applied to every single image in the dataset before training and testing. To a human, the permuted digit is unrecognizable as a "2" — its pixels have been rearranged with no regard for which ones used to be neighbours. The permutation is fixed once and applied identically everywhere, so both models get a full, consistent training set to learn from; the only question is whether either model's own internal computation depends on the original spatial layout.*

![Permutation test results](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_permutation_test.png)

*Mean of 3 seeds, band shows min–max. Both models still learn something from the scrambled pixels — this is an easy 10-class problem and even a poor set of features can separate the classes reasonably well — but the pattern predicted by the theory is visible: the CNN's gap between the solid (normal) and dashed (permuted) lines is consistently wider than the MLP's.*

| Model | Normal-pixel accuracy | Permuted-pixel accuracy | Accuracy drop |
|---|---|---|---|
| MLP (1,810 params) | 97.04% | 96.89% | **0.15 points** (0.2% relative) |
| CNN (810 params) | 98.07% | 93.93% | **4.15 points** (4.2% relative) |

The CNN's accuracy drop is roughly **28× larger in absolute percentage points** than the MLP's, despite the CNN having *fewer* total parameters (810 vs. 1,810) — so the difference is not explained by model capacity. This is measured evidence, not just a plausible-sounding story: the convolutional layer's usefulness depends on the input actually being spatially organised, while the fully-connected layer's does not care either way. It is worth being honest about scale here — on this small, easy 8×8 digit task the effect is real but modest, not a dramatic collapse; on larger, harder image problems with more conv layers stacked (where each layer's usefulness compounds on the layer before it having received spatially meaningful input), the gap would be expected to widen substantially.

## 5. Experiment: from edges to a trained filter bank

The video's intuition section claims CNNs "start with primitive features like edges." This is directly checkable in two ways: first with hand-specified kernels, then with what a CNN trained by plain gradient descent actually discovers on its own.

![Hand-specified edge kernels](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_hand_edge_kernels.png)

*Real 2-D convolution (the exact operation from Section 3) applied to a synthetic test image containing a square, a brighter inner square, and a circle — built at higher resolution than the 8×8 digits specifically so the output is legible. The vertical-edge kernel lights up only at vertical edges (the left and right sides of the squares); the horizontal-edge kernel lights up only at horizontal edges (the tops and bottoms); the Sobel variants combine both, weighting the centre row/column more heavily. This is precisely the kind of orientation-selective response Hubel and Wiesel found in individual neurons — covered in Section 7 — reproduced here with a hand-picked numerical kernel rather than a biological neuron.*

Those kernels were chosen by hand to demonstrate the concept. The more important question is what a CNN *discovers on its own* when trained with no hand-tuning at all — using the identical from-scratch CNN from Section 4, trained on the normal (unpermuted) digits for 40 epochs by plain SGD:

![Learned filters](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_learned_filters.png)

*The CNN reached 97.6% test accuracy. Shown here are its own learned first-layer filters (top row) — nobody specified these numbers; gradient descent found them — and the resulting feature maps on a real test digit (bottom row), after the ReLU nonlinearity. Filters 1 and 2 developed a clear directional structure (a strong colour gradient across the kernel, functionally similar to the hand-specified edge kernels above), and their feature maps show a sparse, localized activation pattern typical of an edge- or gradient-sensitive filter. Filters 3 and 4 stayed closer to flat, contributing comparatively little at this layer for this particular input. This is authentic evidence for the "starts with edges" claim — not asserted, but directly observed in a network's actual learned weights.*

## 6. The biological inspiration: the visual cortex

CNNs are not named after convolution for arbitrary reasons — the architecture is a direct, if simplified, computational echo of how signals are known to move through the mammalian visual system.

![The visual pathway](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/08_visual_pathway.png)

*Light entering the eye is converted to electrochemical signals at the retina. These signals travel along the optic nerve to the thalamus (specifically its lateral geniculate nucleus, LGN), which performs some pre-processing before relaying the signal onward to the primary visual cortex (V1) at the back of the brain. From V1, processing continues into higher visual areas (V2, V4, and beyond) that build increasingly complex representations. The famous experiments described next were recordings taken directly from single cells at the V1 stage.*

## 7. The Hubel and Wiesel cat experiments

In 1959, David Hubel and Torsten Wiesel began recording the activity of individual neurons in the visual cortex of anesthetized cats, projecting patterns of light onto a screen in front of the animal while monitoring which patterns made a given neuron fire. Their original stimuli were simple spots of light — as had already been used successfully for studying earlier stages of the visual pathway — but individual cortical neurons responded only weakly to spots. According to a widely repeated account, one of the key breakthroughs was serendipitous: a neuron fired vigorously not in response to a projected spot, but to the edge of the glass slide itself as it was being removed from the projector, casting a moving straight-edge shadow onto the screen.

Following up on this observation, Hubel and Wiesel found that cortical neurons respond strongly and selectively to bars and edges of light at *specific orientations*. Rotate the bar of light away from a given neuron's preferred angle, and its response falls off — each neuron effectively acts as a detector tuned to one particular edge orientation at one particular location in the visual field. This work was published in 1959 and, with the classification of cell types described next, in a landmark 1962 paper. Hubel and Wiesel were awarded the 1981 Nobel Prize in Physiology or Medicine (shared with Roger Sperry) for this line of research.

## 8. Simple cells, complex cells, and why CNNs are shaped the way they are

Hubel and Wiesel's 1962 paper distinguished two major categories of cells in the visual cortex, and the distinction maps almost directly onto the two core layer types of a CNN.

**Simple cells** respond to an edge of a specific orientation at a specific, narrow location in the visual field — move the edge even a little, or rotate it, and the response drops. **Complex cells** are less strict: they retain a preferred orientation but respond to a bar at that orientation across a range of nearby positions, not just one. Hubel and Wiesel's own interpretation was that complex cells receive pooled input from several simple cells that share an orientation preference but differ slightly in position — an early, biological description of the same idea a modern max-pooling or average-pooling layer implements computationally.

![Simple vs complex cells, and their CNN analogue](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/09_simple_complex_cells.png)

*The direct analogy: a convolutional layer, with its single shared kernel evaluated at every position, behaves like a bank of simple cells all tuned to the same pattern but differing only in which position they're "looking at." A pooling layer, which summarizes several nearby convolutional responses into one, behaves like a complex cell integrating several simple cells with the same orientation preference but slightly different receptive fields — which is also why pooling makes a network's output less sensitive to small translations of the input.*

## 9. From biology to a computer: Neocognitron, LeNet, AlexNet

![Timeline](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/10_timeline.png)

The path from Hubel and Wiesel's biological findings to modern CNNs ran through three major milestones:

**The Neocognitron (Fukushima, 1980)** was the first computational model built explicitly on this biology. It used two named cell types directly modelled on the neuroscience: S-cells ("simple"), which apply a shared 2-D weight pattern at every location in the input (precisely a convolution, though it predates that framing in a machine-learning context), and C-cells ("complex"), whose response is a pooled, nonlinear combination of several S-cells from the same feature type at nearby locations. It was trained with an unsupervised learning algorithm — there was no backpropagation-driven training of the kind used today.

**LeNet-5 (LeCun, Bottou, Bengio & Haffner, 1998)** is widely credited as the first CNN trained end-to-end with supervised gradient-based learning (backpropagation), rather than hand-designed or unsupervised feature detectors. Applied to handwritten digit recognition (the same general task as the digits used in Sections 4–5 of this document, though on the full-size, real-world MNIST-style images LeNet-5 targeted), it saw practical deployment reading digits on bank cheques.

**AlexNet (Krizhevsky, Sutskever & Hinton, 2012)** scaled the same basic architectural ideas — convolutional layers, pooling, and fully-connected layers at the end — to a much larger and deeper network trained on GPUs, and entered the 2012 ImageNet Large Scale Visual Recognition Challenge (ILSVRC). It won by a wide margin over the next-best (non-CNN) approach, and is generally regarded as the result that triggered the shift of the entire field of computer vision toward deep learning.

## 10. Receptive fields: how depth lets a network see more

One direct, quantifiable consequence of stacking convolutional layers is that each successive layer's units can be influenced by a progressively larger region of the original input — its **receptive field** — even though every individual kernel stays small. For stride-1, $k \times k$ convolutions with no pooling, stacking $n$ such layers gives a receptive field of

$$\text{RF}(n) = 1 + n\,(k - 1)$$

![Receptive field growth](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/11_receptive_field.png)

*Real arithmetic for $k=3$: a single layer's output unit is influenced by a 3×3 patch of the input; stack seven such layers and a single output unit is influenced by a 15×15 patch — without ever using a kernel larger than 3×3. This is the computational mechanism behind the simple-cell-to-complex-cell progression described in Section 8: depth, not kernel size, is what lets later layers respond to progressively larger and more complex structures built out of the small, local patterns the earliest layers detect.*

## 11. Applications

The same convolutional building blocks described above are the basis for a wide range of practical computer vision tasks:

- **Image classification** — assigning a single label to an entire image (the task in every experiment in this document).
- **Object detection** — localizing and classifying multiple objects within a single image, typically by predicting bounding boxes alongside class labels.
- **Face recognition** — a specialized form of classification/verification applied to facial images.
- **Image segmentation** — classifying every individual pixel in an image, rather than the image as a whole, to produce a pixel-precise map of where each object is.
- **Image restoration** — using learned convolutional features to plausibly reconstruct old, damaged, or low-resolution photographs.

Each of these builds on exactly the two ideas established in this document: a small, shared kernel scanning the whole image (Sections 2–3), and layered depth building up from simple, local features to progressively larger and more complex ones (Sections 8–10).

## 12. Key takeaways

1. **Flattening an image for a plain ANN is not a neutral preprocessing step.** It causes measurable parameter explosion (Section 2: up to 112,348× more parameters than an equivalent conv layer at 512×512 resolution) and discards information about which pixels are spatially adjacent (Section 4).
2. **A convolution is one small, shared kernel evaluated at every position.** This single property — weight sharing across space — is the entire mechanism behind the parameter-count difference; nothing else needs to change to get it.
3. **Spatial adjacency is not just a theoretical nicety — it is measurably used by a CNN and measurably not used by an MLP.** A fixed pixel permutation applied identically to every image dropped CNN accuracy by 4.15 points and MLP accuracy by only 0.15 points, a 28× difference in absolute terms, despite the CNN having fewer parameters.
4. **"CNNs learn edge detectors first" is not just an analogy — it is directly observable.** A CNN trained with no hand-tuning at all, purely by SGD, developed first-layer filters with clear directional/gradient structure, functionally similar to the hand-specified Sobel-style kernels tested alongside them.
5. **The architecture is a direct computational descendant of a real biological discovery.** Hubel and Wiesel's 1959–1962 recordings from cat V1 identified simple cells (orientation- and position-specific edge detectors) and complex cells (pooled, position-tolerant versions of the same), which map respectively onto a CNN's convolutional and pooling layers.
6. **The line from biology to modern deep learning runs through three concrete milestones**: Fukushima's Neocognitron (1980, unsupervised, biologically literal), LeCun's LeNet-5 (1998, the first backprop-trained CNN), and Krizhevsky, Sutskever & Hinton's AlexNet (2012, the result that triggered the modern deep learning era).
7. **Depth, not kernel size, is what lets a network "see" large-scale structure.** Stacking seven 3×3 conv layers gives the same 15×15 receptive field as one 15×15 kernel, while using far fewer parameters per layer — this is the computational reason a hierarchy of small operations can build up to recognizing whole objects.

## 13. Further reading

- **Hubel, D. H., & Wiesel, T. N. (1959).** *Receptive fields of single neurones in the cat's striate cortex.* The Journal of Physiology, 148(3), 574–591. — The original single-cell recording paper.
- **Hubel, D. H., & Wiesel, T. N. (1962).** *Receptive fields, binocular interaction and functional architecture in the cat's visual cortex.* The Journal of Physiology, 160(1), 106–154. — Introduces the simple cell / complex cell distinction described in Section 8.
- **Fukushima, K. (1980).** *Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position.* Biological Cybernetics, 36(4), 193–202. — The S-cell/C-cell model described in Section 9.
- **LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).** *Gradient-based learning applied to document recognition.* Proceedings of the IEEE, 86(11), 2278–2324. — Introduces LeNet-5, covered in [lenet.md](lenet.md).
- **Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).** *ImageNet classification with deep convolutional neural networks.* Advances in Neural Information Processing Systems (NeurIPS) 25. — The AlexNet paper, covered in [alexnet.md](alexnet.md).
- **Zeiler, M. D., & Fergus, R. (2014).** *Visualizing and Understanding Convolutional Networks.* ECCV 2014. — A much larger-scale version of the learned-filter visualization in Section 5, on real trained ImageNet networks.
- **Karpathy, A.** *CS231n: Convolutional Neural Networks for Visual Recognition* (Stanford course notes, cs231n.github.io). — A widely used, freely available reference covering the architectural details (padding, stride, pooling) that this introductory document sets up but does not itself cover in depth.

---

*Part of an ongoing deep learning notes series. Previous: [adagrad.md](adagrad.md). Next: [lenet.md](lenet.md).*
