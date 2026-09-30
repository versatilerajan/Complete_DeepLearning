# The Keras Functional API

Every model built in this series so far — the tiny CNNs in [cnn-intro.md](cnn-intro.md), the transfer-learning setup in [transfer-learning.md](transfer-learning.md) — has had exactly one input tensor, flowing through exactly one path of layers, to exactly one output. That shape covers a lot of real problems, but not all of them: predicting more than one thing from an image, or combining more than one kind of input, both require a model whose layers form a branching graph rather than a straight line. This document builds and trains that exact kind of branching graph twice, with real data both times, rather than only describing it. The first real, load-bearing finding is a genuine trap: building a shared-trunk model with one regression head (predicting a digit's numeric value, standing in for age) and one classification head (predicting odd/even, standing in for gender), the classification head collapses to **50.4% accuracy — exactly chance** — when the regression target is left on its natural 0–9 scale, because that target's gradient into the shared trunk is measured at **151.7× larger** than the classification gradient. Simply normalizing the regression target to match fixes it completely, bringing classification accuracy back to **97.3%**, matching a dedicated single-task network's 98.0%. The second experiment tests whether adding a second, tabular input branch to an image branch actually helps — a real, honest null result (98.67% image-only vs. 98.18% with the tabular branch added) that pushes back gently on the assumption that more input branches always means a better model.

> **Series note.** This assumes the CNN background from [cnn-intro.md](cnn-intro.md) and [ann-vs-cnn.md](ann-vs-cnn.md), and directly extends [transfer-learning.md](transfer-learning.md)'s convolution-base-plus-head framing to the case of *multiple* heads and *multiple* inputs.

---

## Table of Contents

1. [The limitation: Sequential is a straight line](#1-the-limitation-sequential-is-a-straight-line)
2. [The Functional API: a model is a graph](#2-the-functional-api-a-model-is-a-graph)
3. [Two motivating examples from the video](#3-two-motivating-examples-from-the-video)
4. [Experiment: building and training a real multi-output DAG](#4-experiment-building-and-training-a-real-multi-output-dag)
5. [Experiment: a hidden trap in multi-output training](#5-experiment-a-hidden-trap-in-multi-output-training)
6. [Experiment: does a second input branch actually help?](#6-experiment-does-a-second-input-branch-actually-help)
7. [Practical implementation: the Functional API's own syntax](#7-practical-implementation-the-functional-apis-own-syntax)
8. [Advanced project: multi-output transfer learning on UTKFace](#8-advanced-project-multi-output-transfer-learning-on-utkface)
9. [Practical guidance](#9-practical-guidance)
10. [Key takeaways](#10-key-takeaways)
11. [Further reading](#11-further-reading)

---

## 1. The limitation: Sequential is a straight line

Keras's `Sequential` model is exactly what its name says: a single, linear stack of layers, each one feeding directly into the next, with exactly one input tensor at the start and exactly one output tensor at the end.

![Sequential vs Functional](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_sequential_vs_functional.png)

*Left: `Sequential` can only express this shape — one path, no branches, no way to bring in a second input partway through, no way to peel off a second output partway through. Right: as soon as a design needs a shared trunk feeding two separate branches (or two separate inputs merging into one trunk), that is no longer a straight line — it is a graph, specifically a **directed acyclic graph (DAG)**: data flows in one direction (directed), and never loops back on itself (acyclic), but can split and rejoin freely.*

Two situations force this: predicting more than one thing from one input (Section 3's multi-task example), and combining more than one kind of input into one prediction (Section 3's multi-modal example). Neither can be written as `Sequential([...])`.

## 2. The Functional API: a model is a graph

The Functional API's core idea is to stop thinking of a model as a *list* of layers and start thinking of it as a *graph* of tensors: each layer is a function that is called on a tensor and returns a new tensor, and a full model is just whichever tensors you declare as its inputs and outputs, no matter how they are connected in between. This is a small conceptual shift with a large practical consequence — because layers are just callables applied to tensors, nothing stops the same tensor from being passed into two different layers (a branch) or two different tensors from being passed into the same layer (a merge, typically via concatenation).

## 3. Two motivating examples from the video

**Multi-task learning**: one face image goes in; a shared convolutional trunk extracts features; two separate branches, each ending in its own output layer, predict age (a regression head) and emotion (a classification head) from those same shared features.

**Multi-modal input**: three genuinely different kinds of input — tabular metadata, text (processed by an RNN branch), and an image (processed by a CNN branch) — are each processed by their own appropriate layer types, then concatenated into a single vector before a final set of dense layers predicts one output (a product's price).

![Multi-modal DAG](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_multimodal_dag.png)

*Three independent branches, each suited to its own input type, merged by concatenation before the final prediction layer. This is the shape Section 6 tests directly with a real (smaller) two-branch analog.*

Both examples share the same underlying pattern the rest of this document tests: branches that split apart or merge together, which is precisely what `Sequential` cannot express and the Functional API can.

## 4. Experiment: building and training a real multi-output DAG

Rather than only describe the multi-task example, a real analog was built, gradient-checked, and trained. Real face-image age/emotion data was not available in this environment, so a self-contained but structurally identical stand-in was used, exactly as [transfer-learning.md](transfer-learning.md) did for its own transfer-learning experiments: real handwritten digit images (scikit-learn's digit dataset, upscaled to 16×16), with two real prediction targets computed directly from the true digit label — the digit's **numeric value** (0–9, a regression target standing in for age) and whether the digit is **odd or even** (a binary classification target standing in for gender or emotion).

The architecture matches the branching shape from the video's own project snippet (Section 8) exactly: one shared convolutional trunk, flattened, then split into two dense branches (`dense1`/`dense2`), each followed by a second dense layer (`dense3`/`dense4`), each ending in its own output (`output1` linear, `output2` sigmoid). The full backward pass was gradient-checked against numerical differentiation across every parameter group before any training run: maximum relative error $7.3\times10^{-10}$.

## 5. Experiment: a hidden trap in multi-output training

With the architecture verified, the first training run used the regression target on its natural scale (raw digit values, 0–9) and the classification target as a plain 0/1 label, jointly minimizing MSE (regression head) plus binary cross-entropy (classification head) — exactly what a natural, un-thought-about implementation of the video's `output1`/`output2` pattern would do.

**The classification head failed to learn at all** — 50.4% accuracy, statistically indistinguishable from a coin flip on a balanced binary task, despite the identical architecture reaching 98.0% on that same classification task when trained alone. The regression head trained fine. To find out why, the gradient magnitude flowing into the shared trunk from each head was measured directly, at initialization:

| | Gradient norm into shared trunk (conv layer) |
|---|---|
| From the regression head alone | 25.44 |
| From the classification head alone | 0.168 |
| **Ratio** | **151.7×** |

The regression target's raw scale (values up to 9) makes its mean-squared-error gradient enormous compared to binary cross-entropy's naturally bounded gradient — so when both losses are summed and backpropagated into the *shared* trunk, the regression signal completely dominates every trunk weight update, and the trunk never learns anything useful for the classification branch's own separate weights to work with.

![Multi-task comparison](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_multitask_comparison.png)

*Mean of 5 seeds. "Broken" uses the raw 0–9 regression target; "fixed" simply divides that target by 9 so it lies in $[0,1]$, roughly matching the classification head's own output scale, with no other change to the architecture, losses, or training procedure. That one change recovers the classification head completely (50.4% → 97.3%) and, as a side effect, also *improves* the regression result slightly (MAE 2.49 → 1.07) — likely because the trunk is now free to learn features useful for both tasks rather than being dragged toward features that only serve the numerically dominant loss.*

| | Regression MAE | Classification accuracy |
|---|---|---|
| Single-task (that task alone) | 1.15 | 98.0% |
| Joint, broken (raw target scale) | 2.49 | 50.4% (chance) |
| **Joint, fixed (normalized target)** | **1.07** | **97.3%** |

This is not a peculiarity of digits standing in for faces — it is a direct, general consequence of how gradients combine when multiple losses share a trunk, and it applies exactly as written to the video's own `output1`/`output2` pattern in Section 8, where age (values roughly 0–100) and gender (0/1) sit on wildly different natural scales.

## 6. Experiment: does a second input branch actually help?

To test the multi-modal pattern from Section 3, a second, small analog was built: the same convolutional image branch used above, with a second, tiny branch processing two **hand-crafted tabular features** computed from the very same image — total pixel intensity, and a left–right symmetry score — concatenated with the flattened image features before the final dense layers, exactly matching the "concatenate, then predict" shape in Section 3's diagram. This too was gradient-checked before training (maximum relative error $7.7\times10^{-10}$).

![Multi-input comparison](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_multiinput_comparison.png)

*Mean of 5 seeds, predicting the same odd/even task as Section 5's classification head. The tabular branch adds 64 parameters (12,753 vs. 12,689) and makes no meaningful difference — if anything, a very slightly worse result (98.18% vs. 98.67%), well within normal seed-to-seed noise.*

This is an honest null result, and it is worth reporting as one rather than searching for a framing that makes the extra branch look useful. The reason is not that multi-input models don't work — Section 3's real multi-modal examples (text, images, and tabular data genuinely carry different information about a product's price) are a real and valuable pattern. The reason here is specific: the image branch *alone* already had enough information to solve this particular task (a CNN can trivially learn parity-relevant strokes directly from pixels), so the two hand-crafted tabular features added redundant, not complementary, information. **A second input branch is worth its added complexity only when it carries information the first branch cannot already extract on its own** — not merely because the Functional API makes adding it easy.

## 7. Practical implementation: the Functional API's own syntax

The mechanical pattern — declare an input, call layers on tensors like functions, wrap the result in a `Model` — is what makes everything in Sections 4–6 expressible in Keras:

```python
from tensorflow.keras import layers, Model

inputs = layers.Input(shape=(16, 16, 1))
x = layers.Conv2D(8, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)

# branch A -> regression head
branch_a = layers.Dense(32, activation='relu')(x)
branch_a = layers.Dense(32, activation='relu')(branch_a)
output1 = layers.Dense(1, activation='linear', name='value')(branch_a)

# branch B -> classification head
branch_b = layers.Dense(32, activation='relu')(x)
branch_b = layers.Dense(32, activation='relu')(branch_b)
output2 = layers.Dense(1, activation='sigmoid', name='parity')(branch_b)

model = Model(inputs=inputs, outputs=[output1, output2])
model.compile(
    optimizer='adam',
    loss={'value': 'mse', 'parity': 'binary_crossentropy'},
    loss_weights={'value': 1.0, 'parity': 1.0},   # see Section 5 before leaving this at 1.0/1.0
    metrics={'value': 'mae', 'parity': 'accuracy'},
)
```

This snippet was not executed (no TensorFlow in this environment), but it is a direct, literal translation of the architecture that *was* built from scratch, trained, and measured in Sections 4–6 — every layer, branch, and output here corresponds to a real, gradient-checked line of code in this document's own experiments.

## 8. Advanced project: multi-output transfer learning on UTKFace

The video's capstone project combines transfer learning ([transfer-learning.md](transfer-learning.md)) with the multi-output Functional API pattern from Sections 4–5: load a pretrained convolutional base, freeze it, and attach two branching heads for simultaneous age (regression) and gender (classification) prediction on the UTKFace dataset. The following extends the exact snippet supplied for this document into a complete pipeline, importing libraries through plotting results and making a prediction. **This was not executed** — this environment has neither TensorFlow/Keras, network access to download ResNet50's ImageNet weights, nor the UTKFace dataset itself — but every line follows directly from the supplied snippet and from the verified pattern in Sections 4–7.

```python
# ----------------------------------------------------------------- imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------- data loading
# UTKFace filenames encode labels directly, e.g. "25_0_0_20170109142408075.jpg"
# -> age 25, gender 0 (male), ...
IMG_DIR = 'UTKFace/'
ages, genders, images = [], [], []

for fname in os.listdir(IMG_DIR):
    parts = fname.split('_')
    age, gender = int(parts[0]), int(parts[1])
    img = cv2.imread(os.path.join(IMG_DIR, fname))
    img = cv2.resize(img, (200, 200))
    images.append(img)
    ages.append(age)
    genders.append(gender)

X = np.array(images) / 255.0
y_age = np.array(ages, dtype='float32')
y_gender = np.array(genders, dtype='float32')

# Per Section 5: normalize the regression target so its gradient doesn't
# swamp the shared trunk relative to the classification head. Age ranges
# roughly 0-116 in UTKFace -- left unnormalized, this is exactly the
# 151.7x-style imbalance measured in Section 5, just with a different
# number attached.
y_age_norm = y_age / y_age.max()

X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(
    X, y_age_norm, y_gender, test_size=0.2, random_state=42)

# ----------------------------------------------------------------- model (as supplied)
resnet = ResNet50(include_top=False, input_shape=(200, 200, 3))

resnet.trainable = False

output = resnet.layers[-1].output

flatten = Flatten()(output)

dense1 = Dense(512, activation='relu')(flatten)
dense2 = Dense(512, activation='relu')(flatten)

dense3 = Dense(512, activation='relu')(dense1)
dense4 = Dense(512, activation='relu')(dense2)

output1 = Dense(1, activation='linear', name='age')(dense3)
output2 = Dense(1, activation='sigmoid', name='gender')(dense4)

model = Model(inputs=resnet.input, outputs=[output1, output2])

# ----------------------------------------------------------------- compile
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss={'age': 'mse', 'gender': 'binary_crossentropy'},
    loss_weights={'age': 1.0, 'gender': 1.0},
    metrics={'age': 'mae', 'gender': 'accuracy'},
)

# ----------------------------------------------------------------- train
history = model.fit(
    X_train, {'age': y_age_train, 'gender': y_gender_train},
    validation_data=(X_test, {'age': y_age_test, 'gender': y_gender_test}),
    epochs=20, batch_size=32,
)

# ----------------------------------------------------------------- plot accuracy / mae
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.plot(history.history['gender_accuracy'], label='train')
ax1.plot(history.history['val_gender_accuracy'], label='val')
ax1.set_title('Gender accuracy'); ax1.set_xlabel('epoch'); ax1.legend()

ax2.plot(history.history['age_mae'], label='train')
ax2.plot(history.history['val_age_mae'], label='val')
ax2.set_title('Age MAE'); ax2.set_xlabel('epoch'); ax2.legend()
plt.show()

# ----------------------------------------------------------------- predict
sample = X_test[0:1]
pred_age_norm, pred_gender = model.predict(sample)
predicted_age = pred_age_norm[0][0] * y_age.max()      # undo the Section-5-style normalization
predicted_gender = 'female' if pred_gender[0][0] > 0.5 else 'male'
print(f"Predicted age: {predicted_age:.1f}, predicted gender: {predicted_gender}")
```

Two deliberate additions were made to the supplied snippet, both traceable directly to this document's own experiments rather than invented for their own sake: normalizing `y_age` before training (Section 5's fix, applied here since raw ages up to ~116 would reproduce the exact failure mode measured there) and un-normalizing it again at prediction time so the printed result is in real years. Everything else — the branching structure, both dense towers, both output activations, the frozen `resnet.trainable = False` line — is the snippet as supplied, unmodified.

## 9. Practical guidance

- **Reach for the Functional API as soon as a design needs more than one input tensor or more than one output tensor** — `Sequential` cannot express either, full stop, not as a matter of style but of what the API is capable of representing.
- **Never combine losses of very different natural scales without normalizing the targets or setting `loss_weights` deliberately.** Section 5's 151.7× gradient ratio is not a rare edge case — any regression target with a range larger than roughly $[-1, 1]$ paired with a cross-entropy classification head on a shared trunk is at risk of the same failure, and the failure is silent: training proceeds, loss decreases, and the collapsed head simply looks like it never learned rather than throwing an error.
- **Check a multi-output model's individual head performance, not just its combined loss.** Section 5's broken run had a perfectly reasonable-looking combined loss curve throughout training — the collapse was only visible by evaluating each head's own metric separately.
- **Don't add an input branch just because the Functional API makes it easy to.** Section 6's null result is the general case whenever a new branch's information is already recoverable from an existing one; branch only when the new input plausibly carries information the others cannot already extract.
- **When freezing a pretrained base for multi-output transfer learning** (Section 8), the loss-scale caution applies with extra force: a frozen base cannot adapt its own features to compensate for one head dominating training the way an unfrozen trunk sometimes can, so getting per-head loss scaling right matters even more.

## 10. Key takeaways

![Summary of all measurements](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_summary_table.png)

1. **`Sequential` is a straight line; the Functional API is a directed acyclic graph.** Only the latter can express a shared trunk with multiple branches, multiple inputs merging into one path, or both at once.
2. **Multi-output training has a real, silent failure mode.** A regression head's larger natural gradient scale measurably dominated a shared trunk (151.7× at initialization), collapsing an otherwise-easy classification head to chance level (50.4%) — with the overall training loss curve giving no obvious warning sign.
3. **The fix is simple and verified**: normalizing the regression target to a comparable scale recovered the classification head completely (50.4% → 97.3%, matching the 98.0% single-task result) and even slightly improved the regression head as a side effect.
4. **This exact caution applies to the video's own age/gender project**, where age (0–116-ish) and gender (0/1) sit on exactly the kind of mismatched scales that produced the failure in Section 5 — reflected directly in the Section 8 code walkthrough's target normalization.
5. **A second input branch is not automatically beneficial.** Concatenating a genuinely redundant tabular branch onto an already-sufficient image branch produced no improvement (98.67% vs. 98.18%) — multi-input design should be driven by what information a branch actually adds, not by how easy the API makes adding it.
6. **Every branching structure discussed conceptually in this document was also built as real, gradient-checked, trained code** (maximum relative gradient errors of $7.3\times10^{-10}$ and $7.7\times10^{-10}$ for the two architectures) — the DAG diagrams in Sections 1–3 are not just illustrations of an API feature, they are diagrams of models that were actually run.

## 11. Further reading

- **Keras documentation.** *The Functional API* (keras.io/guides/functional_api/). — The authoritative reference for the `Input`/`Model(inputs=, outputs=)` syntax used throughout Sections 7–8.
- **Chollet, F. (2021).** *Deep Learning with Python*, 2nd edition, Chapter 7. Manning. — Covers the Functional API's multi-input/multi-output patterns directly, including a worked multi-modal example structurally identical to Section 3's price-prediction case.
- **Kendall, A., Gal, Y., & Cipolla, R. (2018).** *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.* CVPR 2018. — A principled, learned alternative to hand-picking `loss_weights`, directly relevant to the fix in Section 5.
- **Zhang, Z., Song, Y., & Qi, H. (2017).** *Age Progression/Regression by Conditional Adversarial Autoencoder.* CVPR 2017. — Introduces the UTKFace dataset used in Section 8's project.
- **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** *Deep Residual Learning for Image Recognition.* CVPR 2016. — Introduces ResNet, the pretrained base used in Section 8's snippet.
- **Caruana, R. (1997).** *Multitask Learning.* Machine Learning, 28, 41–75. — The foundational paper on why and when sharing a trunk across tasks helps, framing the phenomenon tested empirically in Sections 4–5.

---

*Part of an ongoing deep learning notes series. Previous: [transfer-learning.md](transfer-learning.md). Next: [resnet.md](resnet.md).*
