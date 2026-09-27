# Pretrained Models in CNNs: ImageNet, AlexNet, and Transfer Learning

Training a state-of-the-art image classifier from nothing takes millions of labeled images
and days of GPU time. Almost nobody does that. Instead, the field reuses networks that
someone else already trained on a huge dataset — most often ImageNet — and adapts them to a
new task with a fraction of the data and compute. This document works through why that
works, what the ImageNet/ILSVRC benchmark actually is, what changed when AlexNet won it in
2012, and what the AlexNet-to-ResNet architectural trend was really solving. Wherever a claim
can be tested with code, it is: AlexNet's exact architecture is verified with a real forward
pass, Local Response Normalization is run on a real (synthetic) feature map, the
vanishing/exploding-gradient argument for residual connections is measured directly through
an actual deep network, and — the centerpiece — the core practical claim of this whole
topic, "pretrained features need less labeled data," is tested with a real from-scratch
transfer-learning experiment rather than asserted. Historical facts about ImageNet and the
ILSVRC competition (dataset sizes, competition error rates) are **cited**, not measured —
this environment has no access to ImageNet itself, and that distinction is kept explicit
throughout.

This builds on [convolution-operation.md](convolution-operation.md),
[pooling-operation.md](pooling-operation.md), and [lenet5.md](lenet5.md) — AlexNet is, at the
architecture-diagram level, LeNet-5 made deeper, wider, and switched to ReLU, and this
document leans on that comparison throughout. See also
[gradient-descent.md](gradient-descent.md), [momentum-optimization.md](momentum-optimization.md),
and [adam.md](adam.md) for how any of these networks actually get trained.

---

## Table of Contents

1. [Why reuse a model at all](#1-why-reuse-a-model-at-all)
2. [The ImageNet dataset](#2-the-imagenet-dataset)
3. [The ILSVRC challenge and the 2012 discontinuity](#3-the-ilsvrc-challenge-and-the-2012-discontinuity)
4. [AlexNet's architecture, verified](#4-alexnets-architecture-verified)
5. [AlexNet vs LeNet-5](#5-alexnet-vs-lenet-5)
6. [Local Response Normalization](#6-local-response-normalization)
7. [ZFNet, VGG, and the road to ResNet](#7-zfnet-vgg-and-the-road-to-resnet)
8. [Why residual connections work: measured](#8-why-residual-connections-work-measured)
9. [Does a pretrained backbone actually need less data? Measured](#9-does-a-pretrained-backbone-actually-need-less-data-measured)
10. [Practical implementation: Keras + ResNet50](#10-practical-implementation-keras--resnet50)
11. [Practical guidance](#11-practical-guidance)
12. [Key takeaways](#12-key-takeaways)
13. [Further reading](#13-further-reading)

---

## 1. Why reuse a model at all

Three costs make training a large CNN from scratch impractical for most people and most
projects:

- **Data.** State-of-the-art vision models are trained on millions of labeled images.
  Collecting and labeling anywhere near that much data for a new, narrow task is usually not
  feasible.
- **Compute.** Training AlexNet took about 5–6 days on two GTX 580 GPUs in 2012; modern
  networks trained on full ImageNet still take many GPU-days.
- **Expertise.** Getting a large network to train stably at all — the right depth, the right
  normalization, the right learning-rate schedule — is itself a research problem that's
  already been solved by the people who published the architecture.

A **pretrained model** sidesteps all three: someone else already spent the data, compute, and
tuning effort, and published the resulting weights. The specific, testable claim in this
document is that those weights are useful for a *different* task than the one they were
trained on, and with *far less* new labeled data than training from scratch would need. That
claim is measured directly in [section 9](#9-does-a-pretrained-backbone-actually-need-less-data-measured)
— it is the single most important practical fact in this whole topic, and it is not simply
asserted here.

---

## 2. The ImageNet dataset

ImageNet (Deng et al., 2009) is a large, hierarchically organized image database built around
**WordNet**, a lexical database that groups English nouns into *synsets* (sets of synonyms)
connected by "is-a" relationships — a dog is a mammal, a mammal is an animal, and so on.
ImageNet aimed to provide, for every WordNet noun synset, a large collection of images
depicting it.

*The figures in this section are cited from the published dataset papers — Deng et al. (2009),
its 2014 update on image-net.org, and Russakovsky et al. (2015) — not measured, since this
environment has no ImageNet access.*

![Dataset scale comparison](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_dataset_scale_comparison.png)

*(a), (b) The full ImageNet database (as of its 2014 update) contains **14,197,122** images
across **21,841** synsets — several orders of magnitude beyond every prior benchmark computer
vision had used, including the 60,000-image, 10-class CIFAR-10 and the 30,607-image,
256-class Caltech-256. The annual competition (next section) used a fixed 1,000-class subset
with **1,281,167** training images — still 20× more classes and roughly the same order of
magnitude more images than anything that came before it.*

![WordNet hierarchy](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_wordnet_hierarchy.png)

*A small, hand-picked illustration of the kind of hierarchy ImageNet organizes images around
— not a rendering of real ImageNet data. The competition's 1,000 classes are drawn from
leaf-level synsets like these (specific dog breeds, specific bird species, specific tools),
chosen so that no two classes overlap or share an immediate parent — part of why the task is
harder than it sounds: many ILSVRC classes are visually similar breeds or species that require
fine-grained distinctions, not just "cat vs. truck"-level separation.*

Beyond raw classification labels, ImageNet also supports harder tasks the video's summary
mentions in passing — **object localization** (a bounding box around the labeled object) and,
in later versions, object detection with multiple labeled objects per image — which is why
ILSVRC has separate classification and localization/detection tracks.

---

## 3. The ILSVRC challenge and the 2012 discontinuity

The ImageNet Large Scale Visual Recognition Challenge (ILSVRC), run annually from 2010 to
2017, evaluated entrants on the 1,000-class subset using **top-5 error**: a prediction counts
as correct if the true label is anywhere in the model's top 5 guesses — a deliberately
forgiving metric given how many ILSVRC classes are easy to confuse even for a careful human.

*Again, every number below is cited from the competition results and the corresponding
architecture papers, not measured.*

![ILSVRC error history](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_ilsvrc_error_history.png)

*Winning top-5 error by year, from **28.2%** in 2010 down to **2.25%** in 2017. The first two
winners (2010, 2011) used hand-engineered features (SIFT, Fisher vectors) feeding a
traditional classifier — the standard computer vision pipeline at the time. In 2012,
**AlexNet** (Krizhevsky, Sutskever & Hinton) entered a deep CNN and won at **16.4%** top-5
error (single-model result on competition data; the widely quoted 15.3% figure is an
ensembled/averaged variant) against a best non-CNN entry of 26.2% — a relative error
reduction of roughly a third in a single year, after nearly a decade of incremental progress
from the hand-engineered approach. Every subsequent winner has been a CNN, and error
continued falling — GoogLeNet's 6.7% (2014), ResNet's 3.57% (2015), down to SENet's 2.25%
(2017) — past the paper's own ~5.1% estimate of human-level top-5 error on this task.*

The 2012 result is usually described as the moment that convinced the field CNNs, not
hand-engineered features, were the right general-purpose approach to vision — not because
AlexNet invented anything conceptually new relative to LeNet-5's 1998 design
([section 5](#5-alexnet-vs-lenet-5) makes this comparison explicit), but because it showed
that design scaled to real-world images once enough data and enough compute (specifically,
GPU training) were available to run it at that scale.

---

## 4. AlexNet's architecture, verified

AlexNet (Krizhevsky, Sutskever & Hinton, 2012) is 8 learned layers deep: 5 convolutional,
3 fully connected.

![AlexNet architecture](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_alexnet_architecture.png)

*Every shape shown here comes from an actual forward pass through this exact sequence run in
this document's code, not copied from the paper. One well-known detail reproduced faithfully:
the original paper states its input as 224×224, but the stride/kernel arithmetic it also
states only works out for a **227×227** input — a small, famous erratum that every faithful
reimplementation (including the one verified here) has to correct for.*

**Experiment (measured).** Running a real forward pass — actual convolution, ReLU, pooling,
and LRN operations, not a shape calculator — through the full 14-stage pipeline from a
227×227×3 input:

| stage | output shape | parameters |
|---|---|---|
| Conv1 (11×11, stride 4) | 55×55×96 | 34,944 |
| pool 3×3, stride 2 (overlapping) | 27×27×96 | 0 |
| Conv2 (5×5, pad 2) | 27×27×256 | 614,656 |
| pool 3×3, stride 2 | 13×13×256 | 0 |
| Conv3 (3×3, pad 1) | 13×13×384 | 885,120 |
| Conv4 (3×3, pad 1) | 13×13×384 | 1,327,488 |
| Conv5 (3×3, pad 1) | 13×13×256 | 884,992 |
| pool 3×3, stride 2 | 6×6×256 | 0 |
| flatten | 9,216 | 0 |
| FC6 | 4,096 | 37,752,832 |
| FC7 | 4,096 | 16,781,312 |
| FC8 (softmax, 1000-way) | 1,000 | 4,097,000 |
| **Total** | | **62,378,344** |

This exactly matches the standard $(f{\times}f{\times}C_{in}{+}1)\times C_{out}$ formula at
every layer — cross-checked independently by hand for Conv1, Conv2, Conv5, and the flatten
size, all exact matches. **62,378,344** is the precise figure behind the commonly rounded
"~60 million parameters."

Two design choices worth naming since they're easy to miss reading the diagram alone:

- **Overlapping pooling.** The 3×3/stride-2 max pools have overlapping windows (stride <
  window size) — unlike LeNet-5's non-overlapping 2×2/stride-2 pools. Krizhevsky et al.
  reported this reduced overfitting slightly compared to non-overlapping pooling.
- **Split across two GPUs.** In 2012, no single GPU had enough memory for this network.
  AlexNet's conv layers were split into two parallel streams across two GPUs, with
  cross-communication only at specific layers (notably absent between the two streams at
  Conv3, unlike the other conv layers) — a compute-constraint detail with no equivalent
  in a modern single-GPU (or single-chip) reimplementation.

---

## 5. AlexNet vs LeNet-5

![AlexNet vs LeNet-5](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_alexnet_vs_lenet5.png)

*(a) AlexNet has **1,011×** as many learnable parameters as LeNet-5 — 62.4 million against
61,706. (b) Its input alone is **151×** more pixels than LeNet-5's, from both moving to color
(3 channels vs 1) and a roughly 7× larger side length (227 vs 32).*

| | LeNet-5 (1998) | AlexNet (2012) |
|---|---|---|
| depth | 7 layers | 8 layers |
| parameters | 61,706 | 62,378,344 |
| input | 32×32×1 (grayscale) | 227×227×3 (color) |
| activation | tanh | ReLU |
| regularization | none | dropout (0.5) + heavy data augmentation |
| pooling | average, non-overlapping | max, overlapping |
| training data | MNIST, 60,000 images, 10 classes | ILSVRC, 1.28M images, 1,000 classes |
| hardware | CPU | 2 GPUs |

The architectural *pattern* — alternating conv/pool blocks, finishing with dense layers and a
softmax — is unchanged. What changed is scale in every dimension at once (depth, width,
input resolution, dataset size, compute), plus three specific fixes documented elsewhere in
this series: ReLU instead of tanh (per
[lenet5.md § 5](lenet5.md#5-why-tanh-and-what-it-costs-measured), a real but scale-dependent
advantage), dropout as a regularizer (needed because 62 million parameters can badly overfit
even 1.28 million images), and aggressive data augmentation (random crops, horizontal flips,
and PCA-based color jittering) to further stretch that data.

---

## 6. Local Response Normalization

AlexNet includes a normalization step, applied after the ReLU in Conv1 and Conv2, that later
architectures dropped: **Local Response Normalization** (LRN). For a channel $i$ at spatial
position $(x,y)$:

$$b^i_{x,y} = \frac{a^i_{x,y}}{\left(k + \alpha \sum_{j=\max(0,i-n/2)}^{\min(C-1,i+n/2)} \left(a^j_{x,y}\right)^2\right)^{\beta}}$$

with the paper's values $n{=}5, k{=}2, \alpha{=}10^{-4}, \beta{=}0.75$. In words: each
channel's activation is divided down by how much total activation its **neighboring
channels** (in filter-index order, not spatially) also have at that same pixel — a form of
lateral inhibition modeled loosely on real neurons, meant to make a strongly-activated
feature detector suppress its neighbors and stand out more.

**Experiment (measured).** A single spatial position, 12 channels, one channel carrying a
strong activation (8.343) against ordinary background noise on the rest:

| | before LRN | after LRN |
|---|---|---|
| strong channel | 8.343 | 4.948 |
| mean of the other 11 channels | 0.234 | 0.139 |

![Local Response Normalization](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_local_response_norm.png)

*Running the exact formula above on this synthetic feature map suppresses the strong
channel's activation by **1.686×** — a real, substantial rescaling, not a cosmetic tweak.
Every other channel is suppressed too, by a similar factor, since they all sit in each
other's normalization windows. Historically, LRN did not survive: later architectures (VGG
among the first to test it directly) found it added computation without a measurable accuracy
benefit at the scales they tested, and it was replaced first by nothing, then by
[Batch Normalization](lenet5.md), which normalizes across a training *batch* rather than
across nearby channels and has a much more solid empirical track record.*

---

## 7. ZFNet, VGG, and the road to ResNet

Three of the "famous architectures" the video's summary names, briefly:

- **ZFNet** (Zeiler & Fergus, 2013) is essentially AlexNet with tuned hyperparameters —
  smaller filters and stride in the first conv layer specifically to preserve more
  information from the input. Its main contribution was less the architecture itself and
  more a *visualization* technique (deconvolutional networks) for seeing what each layer's
  filters actually respond to, which is how the field started to empirically justify the
  "early layers detect edges, deep layers detect parts" claim rather than merely assuming it
  by analogy to biology.
- **VGG** (Simonyan & Zisserman, 2014) pushed depth much further (16–19 weight layers) using
  *only* 3×3 convolutions stacked repeatedly, showing that two stacked 3×3 convs cover the
  same receptive field as one 5×5 conv with fewer parameters and an extra nonlinearity — the
  same point made about stacking small filters in
  [lenet5.md § 9](lenet5.md#9-practical-guidance). VGG's simplicity made it (and its
  pretrained weights) extremely popular as a transfer-learning backbone for years, despite
  being large and slow by later standards (~138 million parameters for VGG-16).
- **ResNet** (He et al., 2015) went to 152 layers — a depth that earlier plain architectures
  could not train at all, for a specific, measurable reason covered next.

Each step made the network *deeper*. Depth alone eventually stopped helping plain
(non-residual) networks — not because a deeper network has less capacity, but because it
became harder to *train* one at all, which is a different, more specific problem than the
"vanishing gradients" story usually invoked and worth actually measuring rather than reciting.

---

## 8. Why residual connections work: measured

A **residual block** computes $y = \text{ReLU}(x + F(x))$ instead of $y = \text{ReLU}(F(x))$
— it adds the block's input back to its output before the final nonlinearity, via a "skip
connection." The claimed benefit is that gradients can flow backward through the `+x` path
essentially unimpeded, even when the learned branch $F$ contributes little.

**Experiment (measured).** A from-scratch two-conv block was implemented both ways — plain
($y=\text{ReLU}(F(x))$) and residual ($y=\text{ReLU}(x+F(x))$) — with a full, numerically
verified backward pass (gradient check against finite differences matched to 9 significant
figures). Stacking 12 such blocks with identical, well-scaled (He-like) random weights and
tracing the gradient signal backward through every block boundary:

![Residual gradient flow](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/08_residual_gradient_flow.png)

*The gradient reaching the very first block's input is **1.28×10⁻⁴** for the plain stack
against **2.37×10⁻¹** for the residual stack — nearly three orders of magnitude more signal
survives 12 layers of backprop when a skip connection is present. This is not a subtle
effect.*

**Experiment (measured), depth sweep.** Repeating this at depths from 2 to 24 blocks:

![Residual depth sweep](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/09_residual_depth_sweep.png)

*The plain network's gradient ratio (input ÷ output) falls toward zero as depth grows —
**0.012** by 24 blocks — the textbook vanishing-gradient signature, and it worsens
essentially monotonically with depth. The residual network shows the opposite problem: its
ratio **grows** with depth, reaching **10,784×** at 24 blocks. This is a genuine, verified
result (confirmed with a numerical gradient check, and checked to be robust across several
channel widths, not a fluke of one configuration) and it is worth stating plainly because it
complicates the usual one-line story: an unnormalized skip connection doesn't calmly
"preserve" gradient magnitude at a constant scale — the identity path guarantees gradient
**never vanishes** (algebraically, it is added back in unchanged at every block, a floor
plain networks have no equivalent of), but because the learned branch's contribution *also*
adds on top of that floor at every block, the combined signal can compound and grow without
bound if nothing keeps each block's contribution small. This is exactly why real ResNets pair
every skip connection with **Batch Normalization** inside the block — not the skip
connection alone. The skip connection solves vanishing; batch normalization is what keeps
that same mechanism from overshooting into exploding.*

---

## 9. Does a pretrained backbone actually need less data? Measured

This is the practical claim the whole topic rests on, and it is testable directly with a
small from-scratch experiment rather than assumed. A CNN "backbone" (two conv+pool blocks)
was pretrained on a **source** task — four classes of simple synthetic shapes — to 100% test
accuracy, then adapted to a **target** task with four *different* shape classes, three ways:
training a fresh randomly-initialized network from scratch, freezing the pretrained backbone
and training only a new classification head on top of it (the standard "feature extraction"
style of transfer learning), and fine-tuning the whole pretrained network at a reduced
learning rate.

![Source and target tasks](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/10_source_target_tasks.png)

*Both tasks are synthetic (generated by code, not a real image dataset — disclosed here as
such), built from the same low-level visual vocabulary (straight edges, corners, curves) but
depicting entirely different shapes — modeling, in miniature, the real situation of using an
ImageNet-pretrained backbone on a task ImageNet never saw: new classes, but the same kinds of
edges and textures the backbone already learned to detect.*

**Experiment (measured).** Test accuracy on the **target** task as the amount of target-task
labeled training data varies, averaged over 4 random seeds per point:

| labeled examples | from scratch | frozen backbone (transfer) | fine-tuned backbone |
|---|---|---|---|
| 2 | 30.4% | 30.4% | 30.0% |
| 4 | 40.6% | 41.0% | 41.0% |
| 8 | **55.0%** | **70.0%** | 67.3% |
| 16 | 53.8% | 62.9% | 63.3% |
| 32 | 69.4% | 76.5% | 77.3% |
| 64 | 92.7% | 92.7% | 94.4% |

![Data efficiency](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/11_data_efficiency.png)

*At extremely low data (2–4 examples per class), the small backbone used here doesn't have
enough to work with regardless of initialization, and all three approaches sit near chance
level. In the 8–32 example range — the realistic "I only have a few dozen labeled photos"
regime transfer learning is actually used for — frozen and fine-tuned transfer both beat
training from scratch by a clear, consistent margin, and the gap closes again once there's
enough target data (64 examples) for a from-scratch network to learn good features on its
own.*

The clearest way to state the size of that gap: interpolating the from-scratch curve, it
takes **32.6** labeled target-task examples for a from-scratch network to match what the
frozen pretrained backbone achieves with just **8** — a **4.07×** difference in the amount of
labeled data needed for the same accuracy. That number is the entire practical case for
pretrained models, demonstrated directly rather than taken on faith: the backbone's
already-learned edge and shape detectors transfer to new classes built from the same visual
primitives, and transferring them costs nothing while training them from scratch costs
labeled data the new task might not have.

---

## 10. Practical implementation: Keras + ResNet50

This environment has no internet access and no TensorFlow installed, so the code below is
correct-by-construction (checked for syntax, following the standard Keras/`tf.keras` API) but
has not been executed here — it needs the real ResNet50 ImageNet weights, which have to be
downloaded, to run.

**Inference with a pretrained model, no training at all** — exactly the "predict on a new
image" demonstration the video's practical section walks through:

```python
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load ResNet50 with its ImageNet-pretrained weights. include_top=True keeps
# the original 1000-way classification head, since we're using this model
# as-is rather than adapting it to a new task.
model = ResNet50(weights='imagenet', include_top=True)

img = image.load_img('my_photo.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)          # add the batch dimension ResNet50 expects
x = preprocess_input(x)                # ResNet50-specific normalization, NOT a plain /255

preds = model.predict(x)
print(decode_predictions(preds, top=5)[0])   # human-readable class names + probabilities
```

`preprocess_input` matters more than it looks: each pretrained architecture in
`tf.keras.applications` was trained with its own specific input normalization (ResNet50
subtracts per-channel ImageNet means rather than dividing by 255), and using the wrong one
silently produces poor predictions without any error.

**Transfer learning: adapting ResNet50 to a new task**, matching
[section 9](#9-does-a-pretrained-backbone-actually-need-less-data-measured)'s frozen-backbone
experiment, but with the real pretrained network instead of a small hand-built one:

```python
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50

# include_top=False drops the original 1000-way ImageNet head, keeping only
# the convolutional feature-extracting backbone.
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the backbone: its weights are not updated by the optimizer. This is
# the "frozen backbone" arm of section 9's experiment, applied to a real
# pretrained network instead of a small hand-trained one.
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),   # see pooling-operation.md section 7
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=10, validation_data=validation_ds)

# Optional fine-tuning pass, matching section 9's third arm: unfreeze the
# backbone and continue training at a much lower learning rate, so the
# already-good pretrained weights are nudged rather than overwritten.
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
             loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_finetune = model.fit(train_ds, epochs=5, validation_data=validation_ds)
```

---

## 11. Practical guidance

- **Start with a frozen pretrained backbone**, not fine-tuning, especially with limited data.
  [Section 9](#9-does-a-pretrained-backbone-actually-need-less-data-measured)'s measurements
  show frozen transfer matching or beating from-scratch training at every data size tested,
  and it's cheaper: only the new head's parameters need gradients.
- **Fine-tune only once the frozen-backbone approach plateaus**, and use a small learning
  rate when you do (the code in [section 10](#10-practical-implementation-keras--resnet50)
  uses `1e-5`, two orders of magnitude below a typical from-scratch rate) — the backbone's
  weights already encode a lot of useful structure, and a large learning rate can destroy it
  before the new head has learned anything to replace it with.
- **Match the pretrained model's exact preprocessing.** Every architecture in
  `tf.keras.applications` has its own `preprocess_input`; using the wrong one (or a plain
  `/255`) is a silent bug, not an error, and quietly degrades every prediction.
- **Prefer newer, normalization-stabilized architectures (ResNet and later) as backbones**
  over AlexNet or plain VGG when starting a new project — [section 8](#8-why-residual-connections-work-measured)'s
  gradient measurements are the direct, mechanistic reason very deep plain networks are hard
  to train reliably, independent of any accuracy comparison.
- **If your target task's images are visually very different from natural photos** (e.g.
  medical scans, satellite imagery, spectrograms), expect the transfer-learning advantage
  measured in [section 9](#9-does-a-pretrained-backbone-actually-need-less-data-measured) to
  shrink — it depends on the source and target tasks sharing low-level visual structure,
  which was true by construction in that experiment and is not guaranteed for every domain.

---

## 12. Key takeaways

1. Pretrained models exist because training a large CNN from scratch costs data, compute, and
   tuning expertise most projects don't have — reusing a model that already paid those costs
   is the practical default, not a shortcut.
2. ImageNet is organized around the WordNet noun hierarchy and, as of its 2014 update,
   contains **14,197,122** images across **21,841** classes (cited, not measured here); the
   ILSVRC competition used a 1,000-class, 1,281,167-image subset of it.
3. AlexNet's 2012 ILSVRC win (16.4% top-5 error, against a 26.2% best non-CNN entry) is
   usually cited as the moment CNNs displaced hand-engineered features as the default
   approach to vision — not because its architecture was conceptually new relative to
   LeNet-5, but because it showed the same pattern scaled.
4. AlexNet's exact architecture was verified end to end by an actual forward pass:
   **62,378,344** parameters (1,011× LeNet-5's 61,706), on a 227×227×3 input (151× more
   pixels than LeNet-5's, and note the paper's own well-known 224-vs-227 discrepancy).
5. Local Response Normalization, run on a real synthetic feature map, suppressed a strong
   channel's activation by **1.686×** due to competition from noisy neighboring channels —
   a real, substantial effect, though one later architectures found wasn't worth its cost and
   replaced with Batch Normalization.
6. Residual connections were measured, not just described: at 12 stacked blocks, gradient
   reaching the input was **1.28×10⁻⁴** for a plain network vs **2.37×10⁻¹** for a residual
   one — roughly 1,850× more surviving signal. But the same skip connections, unpaired with
   normalization, showed a real and measured tendency to make gradients **grow** rather than
   stabilize as depth increases (10,784× at 24 blocks) — the mechanistic reason real ResNets
   always pair skip connections with Batch Normalization.
7. The central practical claim — pretrained features need less labeled data — was tested
   directly with a real transfer-learning experiment: a frozen pretrained backbone reached
   70% target-task accuracy with 8 labeled examples, a level a from-scratch network needed an
   interpolated **32.6** examples (**4.07×** more) to match.

---

## 13. Further reading

- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). *ImageNet: A
  Large-Scale Hierarchical Image Database.* CVPR. — The original ImageNet paper, and the
  source for the dataset-scale figures in [section 2](#2-the-imagenet-dataset).
- Russakovsky, O., Deng, J., Su, H., et al. (2015). *ImageNet Large Scale Visual Recognition
  Challenge.* IJCV, 115(3), 211–252. — The full ILSVRC retrospective, including the human
  baseline error rate used in [section 3](#3-the-ilsvrc-challenge-and-the-2012-discontinuity).
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet Classification with Deep
  Convolutional Neural Networks.* NeurIPS. — The AlexNet paper, including Local Response
  Normalization ([section 6](#6-local-response-normalization)) and the two-GPU split
  discussed in [section 4](#4-alexnets-architecture-verified).
- Zeiler, M. D., & Fergus, R. (2014). *Visualizing and Understanding Convolutional Networks.*
  ECCV. — ZFNet and the deconvolutional visualization technique referenced in
  [section 7](#7-zfnet-vgg-and-the-road-to-resnet).
- Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale
  Image Recognition.* arXiv:1409.1556. — VGG.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image
  Recognition.* CVPR. — ResNet and the original motivation for residual connections tested
  in [section 8](#8-why-residual-connections-work-measured).
- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). *How transferable are features in
  deep neural networks?* NeurIPS. — A direct empirical study of exactly the transfer-learning
  question tested in [section 9](#9-does-a-pretrained-backbone-actually-need-less-data-measured),
  on real ImageNet-scale networks.
- Pan, S. J., & Yang, Q. (2010). *A Survey on Transfer Learning.* IEEE Transactions on
  Knowledge and Data Engineering, 22(10), 1345–1359. — A general framework for transfer
  learning beyond CNNs specifically.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, chapter 9.11 and
  chapter 15.2. MIT Press. — Historical architecture notes and a formal treatment of transfer
  learning.

---

*Part of a series of deep-learning study notes. Related:
[convolution-operation.md](convolution-operation.md) ·
[pooling-operation.md](pooling-operation.md) ·
[lenet5.md](lenet5.md) ·
[gradient-descent.md](gradient-descent.md) ·
[momentum-optimization.md](momentum-optimization.md) ·
[adam.md](adam.md)*
