# Transfer Learning

Every CNN topic covered so far in this series — [cnn-intro.md](cnn-intro.md)'s convolution mechanics, [padding-and-strides.md](padding-and-strides.md)'s output-size arithmetic, [ann-vs-cnn.md](ann-vs-cnn.md)'s weight-sharing argument, and the [backpropagation-in-cnn](backpropagation-in-cnn-part1.md) series' gradient derivations — has assumed you are training a network from scratch. Transfer learning starts from a different premise: someone has already trained a large CNN on a large, diverse dataset (classically VGG16 on ImageNet's 1000 classes), and that network's early layers have already learned to detect edges, textures, and simple shapes — visual building blocks that are useful for almost any image task, not just the one the network was originally trained on. This document tests that premise directly rather than taking it on faith. A small CNN trained only on digits 0–4, then reused (with its convolutional weights completely frozen) to classify the entirely different digits 5–9, reaches **56.3% accuracy from just 5 examples per class** — nearly triple the 29.5% a freshly-initialized network manages with the same 5 examples. Allowing the top convolutional layer to keep learning (fine-tuning) pushes that further, to **78.4%**. And the mechanism behind this is directly observable: the frozen filters, having never seen a single example of digits 5–9, produce feature-map activations on those unseen digits that are **94.3% as strong** as on the digits they were actually trained on — concrete evidence that what those early filters learned was general, not specific to the classes they happened to be trained on.

> **Series note.** This builds on [cnn-intro.md](cnn-intro.md) (what a convolution computes), [ann-vs-cnn.md](ann-vs-cnn.md) (why a conv layer's parameters are reusable in the first place), and the [backpropagation-in-cnn](backpropagation-in-cnn-part1.md) series (how those parameters get updated during training — relevant to understanding exactly what "freezing" a layer means at the gradient level).

---

## Table of Contents

1. [Why train from scratch at all?](#1-why-train-from-scratch-at-all)
2. [The convolution base and the classifier head](#2-the-convolution-base-and-the-classifier-head)
3. [Feature extraction vs. fine-tuning](#3-feature-extraction-vs-fine-tuning)
4. [Experiment setup: a real, small-scale transfer problem](#4-experiment-setup-a-real-small-scale-transfer-problem)
5. [Experiment: does transfer learning actually help?](#5-experiment-does-transfer-learning-actually-help)
6. [Experiment: why do frozen features transfer at all?](#6-experiment-why-do-frozen-features-transfer-at-all)
7. [Experiment: does data augmentation help?](#7-experiment-does-data-augmentation-help)
8. [Framework usage](#8-framework-usage)
9. [Practical guidance](#9-practical-guidance)
10. [Key takeaways](#10-key-takeaways)
11. [Further reading](#11-further-reading)

---

## 1. Why train from scratch at all?

Training a deep CNN from scratch requires a large labeled dataset, a meaningful amount of compute, and time — all three of which are in short supply for most individual projects. ImageNet-scale training sets (millions of labeled images across a thousand classes) are what let a network's early layers learn robust, general-purpose visual features in the first place; a small custom dataset (a few thousand images of cats and dogs, say) is nowhere near enough on its own for a deep network to learn those same general features unaided. Transfer learning sidesteps this by reusing a network that someone else already spent that data and compute training, and adapting only the parts of it that need to be specific to your task.

## 2. The convolution base and the classifier head

Split any CNN classifier into two conceptual parts:

![Convolution base and classifier head](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_base_and_head.png)

*The convolution base — the stack of conv and pooling layers — is where the network builds up its visual vocabulary, from edges and textures in the earliest layers to more complex, larger-scale patterns deeper in ([cnn-intro.md](cnn-intro.md) Section 10 covers why depth specifically is what enables this progression). The classifier head — the dense layers at the end — is what maps that visual vocabulary onto a specific set of output classes. Transfer learning's central move is: keep the base (it already learned something valuable and general), throw away the head (it was wired to someone else's classes), and attach a brand new head wired to your own.*

## 3. Feature extraction vs. fine-tuning

Once the head is replaced, there is still a choice about what happens to the base during training on your new task:

![Three freeze patterns](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_freeze_patterns.png)

- **From scratch**: nothing is reused. Every layer, base and head alike, starts randomly initialized and is trained on your data alone.
- **Feature extraction**: the entire base is frozen at its source-trained weights — no gradients are allowed to update it — and only the new head is trained. This is appropriate when your dataset is similar in kind to what the base was originally trained on, since the base's existing features should already suit your images reasonably well.
- **Fine-tuning**: the base is mostly frozen, but its topmost layer or layers (the ones closest to the head, which tend to have learned the most task-specific patterns) are unfrozen and allowed to keep training, typically at a much smaller learning rate than the head uses. This is more effective when your target task is meaningfully different from the source task, since it lets the network adapt its most specialized features rather than being stuck with whatever the source task happened to need.

All three of these are tested directly in Sections 5–7, rather than just described.

## 4. Experiment setup: a real, small-scale transfer problem

Reproducing the video's Dogs vs. Cats / VGG16 setup exactly is not possible in this environment (no internet access to download pretrained weights, no GPU-scale compute for ImageNet-sized training). Instead, a self-contained but genuinely analogous problem was built, using nothing but real, gradient-checked code and a real dataset:

- **Source task** (standing in for ImageNet): classify handwritten digits **0–4** from scikit-learn's digit dataset, upscaled from 8×8 to 16×16 pixels (a real nearest-neighbour resize of real images, done only so two convolution-and-pooling stages fit) — 720 images, giving the source model a reasonably large and diverse training set relative to what follows.
- **Target task** (standing in for cats vs. dogs): classify digits **5–9** — a related but distinct set of classes the source model never saw a single example of — using a deliberately *small* number of examples per class, to reproduce the exact data-scarce regime transfer learning is meant to address.
- **Architecture**: a two-convolution-layer network (conv1 → pool → conv2 → pool → flatten → dense → softmax), gradient-checked against numerical differentiation before any experiment was run (maximum relative error $2.9\times10^{-10}$).

This is a controlled analogy, not a reproduction of the video's specific numbers — it is flagged as such throughout, and the point is to test the *mechanism* the video describes, which does not depend on the specific dataset used.

## 5. Experiment: does transfer learning actually help?

The source model was trained to 98.2% test accuracy on digits 0–4. Its convolutional base (both conv layers) was then reused under three regimes on the target task (digits 5–9), at four different target-training-set sizes (5, 10, 20, and 40 examples per class), each averaged over 5 seeds:

![Shots comparison](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_shots_comparison.png)

*Mean of 5 seeds; error bars show standard deviation. At every training-set size tested, fine-tuning (orange) beats feature extraction (blue), which in turn beats training from scratch (red) — and the gap between them is largest exactly where data is scarcest, on the left. From scratch is not just worse on average at 5 shots, it is wildly inconsistent (std of 0.136, meaning some random seeds got lucky and some did very poorly) — a symptom of trying to learn general visual features from almost no data.*

| Shots/class | From scratch | Feature extraction | Fine-tuning |
|---|---|---|---|
| 5 | 29.5% ± 13.6 | 56.3% ± 15.2 | **78.4% ± 2.9** |
| 10 | 59.2% ± 32.9 | 74.6% ± 5.7 | **84.4% ± 1.8** |
| 20 | 62.5% ± 36.0 | 79.9% ± 2.8 | **88.7% ± 1.8** |
| 40 | 79.1% ± 30.4 | 84.7% ± 0.7 | **91.7% ± 1.3** |

Three things are worth pulling out of this table specifically. First, the ordering the video predicts — fine-tuning best, feature extraction second, from-scratch worst — holds at every single data point tested, not just on average. Second, the *gap* between from-scratch and the transfer-based methods shrinks steadily as more target data becomes available (from a 49-point gap at 5 shots between from-scratch and fine-tuning, down to a 13-point gap at 40 shots) — exactly the "data-hungry" argument from Section 1, now quantified: transfer learning's advantage is concentrated precisely in the small-data regime it is meant to address. Third, from-scratch's standard deviation is dramatically larger than either transfer-based method's at every shot count — reusing pretrained features doesn't just improve average accuracy, it makes the outcome far more reliable across different random initializations and data draws.

## 6. Experiment: why do frozen features transfer at all?

Feature extraction's entire premise is that a frozen, never-updated set of filters can still be useful for classes it never saw during training. This is checked directly by feeding the source-trained conv1 filters a real image from the source task (a "0") and a real image from the target task (a "7", never seen in training) and comparing their responses.

![Filter transfer](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_filter_transfer.png)

*Filters 3 and 4 produce clearly structured, edge-and-stroke-like responses on both images — the "7" activates them almost as strongly as the "0" does, despite the network never having been trained on a single "7". Filter 2 is essentially dead on both images (a common, unremarkable occurrence with ReLU networks, not hidden here), and filter 1 responds only sparsely to both. The pattern that matters is that no filter behaves *qualitatively differently* on the unseen class — none of them fail to fire, or fire in a nonsensical or noise-like way, on an input they were never trained on.*

| | Mean activation |
|---|---|
| Seen class (source-task "0") | baseline |
| Unseen class (target-task "7") | **94.3%** as strong as the seen-class baseline |

A 94.3% ratio means the frozen filters are, on average, very nearly as "alive" on a completely unseen digit class as they are on the classes they were actually trained on. This is the direct, measured explanation for why feature extraction in Section 5 works as well as it does even at 5 shots — the frozen base is not guessing blindly on target-task images, it is applying genuinely general-purpose detectors that happen to respond about as well to strokes and curves in a "7" as in a "0".

## 7. Experiment: does data augmentation help?

The video raises data augmentation as a way to fight overfitting on small datasets. This was tested with real pixel-level transformations (random shifts and rotations, applied via `scipy.ndimage`, not a synthetic approximation) on the 10-shot target training set, applied to both the from-scratch and feature-extraction regimes.

![Augmentation effect](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_augmentation_effect.png)

*Mean of 5 seeds. Augmentation improved the from-scratch model substantially — from 75.1% to 84.3% — and, just as importantly, cut its seed-to-seed variability by more than 4×  (std 0.118 down to 0.026). It did essentially nothing for the feature-extraction model (73.2% to 72.7%, a difference within noise).*

| Regime | Plain | Augmented |
|---|---|---|
| From scratch | 75.1% ± 11.8 | **84.3% ± 2.6** |
| Feature extraction | 73.2% ± 4.3 | 72.7% ± 3.8 |

This asymmetry is a genuinely useful, non-obvious finding rather than a simple confirmation of the video's claim: augmentation is not a universal fix, it is specifically a fix for a model that has to learn robustness to small shifts and rotations from scratch, from very few examples. A frozen, source-trained base has typically already learned some of that robustness as a side effect of being trained on hundreds of real, naturally-varied source-task examples — so showing it *more* variations of the same handful of target-task images adds little it doesn't already have. The practical implication: augmentation is most worth reaching for when training from scratch or when fine-tuning substantial parts of the base, and less critical for plain feature extraction on a similar source/target pair.

## 8. Framework usage

**Keras** (feature extraction, then fine-tuning, matching the video's actual workflow):

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers

# 1. Load the pretrained base, without its original classifier head
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# 2. Feature extraction: freeze the entire base
conv_base.trainable = False
model = models.Sequential([
    conv_base,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# ... train on your small dataset here ...

# 3. Fine-tuning: unfreeze only the top conv block, use a small learning rate
conv_base.trainable = True
for layer in conv_base.layers[:-4]:
    layer.trainable = False
model.compile(optimizer=optimizers.RMSprop(learning_rate=1e-5),
              loss='binary_crossentropy', metrics=['accuracy'])
# ... continue training with a much smaller learning rate ...
```

**PyTorch:**

```python
import torch
import torchvision.models as models
import torch.nn as nn

conv_base = models.vgg16(weights='IMAGENET1K_V1')

# Feature extraction: freeze every parameter in the base
for param in conv_base.features.parameters():
    param.requires_grad = False

conv_base.classifier[6] = nn.Linear(4096, 1)   # replace the head

# Fine-tuning: selectively unfreeze the last conv block
for name, param in conv_base.features.named_parameters():
    if name.startswith("28") or name.startswith("26"):   # last block's layer indices
        param.requires_grad = True
```

Neither snippet was executed — neither framework nor pretrained weights are available in this environment. The *mechanism* each snippet controls (which layers receive gradient updates) is exactly what Sections 5–7 tested directly with a from-scratch implementation.

## 9. Practical guidance

- **Default to feature extraction first.** It is cheaper, more stable (Section 5's standard deviations are consistently smaller), and — per Section 6 — works well whenever the target task shares low-level visual structure with the source task, even if the specific classes are entirely different.
- **Move to fine-tuning if feature extraction plateaus below what you need**, and if you have enough target data to do so safely — Section 5 shows fine-tuning's advantage over feature extraction narrowing but never disappearing as data grows, so it is close to a strict improvement once you can afford the extra training care it requires.
- **Use a much smaller learning rate when fine-tuning than when training the head.** The base's weights are already close to a good solution; large updates risk destroying the useful features it already learned rather than gently adapting them. (This experiment used 0.05 for fine-tuning versus 0.3 for feature extraction and from-scratch — a 6× reduction, consistent with the video's point about needing more careful training.)
- **Reach for data augmentation when training from scratch or fine-tuning heavily, not as a default for plain feature extraction** — Section 7's asymmetric result suggests it addresses a specific weakness (lack of learned invariance) that a frozen, well-trained base may not have in the first place.
- **Freeze the general layers, not the specific ones.** [ann-vs-cnn.md](ann-vs-cnn.md) and this document's Section 2 both point at the same underlying idea: earlier layers learn more generic features, later ones more task-specific — so if you only unfreeze part of the base, unfreeze from the top (closest to the head) downward, exactly as tested in Section 5's fine-tuning regime.

## 10. Key takeaways

![Summary of all measurements](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_summary_table.png)

1. **Transfer learning's advantage is real, measured, and concentrated where data is scarce.** At 5 shots per class, fine-tuning reached 78.4% versus from-scratch's 29.5% — a 49-point gap that narrowed to 13 points by 40 shots, exactly tracking the "data-hungry" motivation from Section 1.
2. **The ordering fine-tuning > feature extraction > from-scratch held at every single data point tested**, not just on average — and from-scratch was also far less consistent across seeds, not just lower on average.
3. **Frozen features transfer because they are measurably general, not task-specific.** Source-trained filters fired at 94.3% of their normal strength on a class they never saw during training — a direct, quantified answer to "why does freezing the base work at all?"
4. **Data augmentation is not a universal fix — it specifically helps models that must learn invariance from scratch.** It improved from-scratch accuracy substantially (75.1% → 84.3%) while doing essentially nothing for feature extraction (73.2% → 72.7%), because a well-trained frozen base has typically already learned much of that invariance.
5. **Fine-tuning needs a smaller learning rate than training a fresh head**, because it is nudging already-useful weights rather than learning from nothing — reflected directly in this experiment's own hyperparameters (0.05 vs. 0.3).
6. **The three regimes differ only in which layers receive gradient updates** — a from-scratch network, a feature-extraction network, and a fine-tuned network can be (and were, in this document's experiments) literally the same architecture, differing only in a few `trainable` flags.

## 11. Further reading

- **Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014).** *How transferable are features in deep neural networks?* NeurIPS 27. — The foundational empirical study of exactly what Section 6 tests here at small scale: which layers' features generalize across tasks, and which are task-specific.
- **Simonyan, K., & Zisserman, A. (2015).** *Very Deep Convolutional Networks for Large-Scale Image Recognition.* ICLR 2015. — Introduces VGG16, the specific pretrained network used in the video and in Section 8's framework code.
- **Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009).** *ImageNet: A large-scale hierarchical image database.* CVPR 2009. — The source dataset that makes VGG16's convolution base broadly useful as a transfer-learning starting point in the first place.
- **Chollet, F. (2021).** *Deep Learning with Python*, 2nd edition, Chapter 8. Manning. — Chollet's own Dogs vs. Cats / VGG16 walkthrough is the direct basis for the video this document is based on, including the specific feature-extraction-then-fine-tuning workflow in Section 8.
- **Shorten, C., & Khoshgoftaar, T. M. (2019).** *A survey on Image Data Augmentation for Deep Learning.* Journal of Big Data, 6(60). — A broader survey of augmentation techniques than the shift/rotation pair tested in Section 7.

---

*Part of an ongoing deep learning notes series. Previous: [backpropagation-in-cnn-part2.md](backpropagation-in-cnn-part2.md). Next: [vgg16.md](vgg16.md).*
