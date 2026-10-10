# Deep (Stacked) RNN, LSTM, and GRU

A single recurrent layer gives a network one level of abstraction over time. Stacking several recurrent layers on top of each other — a **Deep RNN**, also called a **Stacked RNN** — lets the network build a hierarchy of representations, the same way stacking convolutional layers does for images. This note covers the architecture, the time/depth notation, why `return_sequences=True` matters when stacking layers in Keras, and real parameter counts for stacked SimpleRNN, LSTM, and GRU models, following directly from [recurrent-neural-networks-intro.md](recurrent-neural-networks-intro.md), [lstm-introduction.md](lstm-introduction.md), and [gru-gated-recurrent-units.md](gru-gated-recurrent-units.md) — everything here applies equally to all three cell types.

## Table of Contents
1. [From One Layer to Many](#1-from-one-layer-to-many)
2. [Notation: Time and Depth](#2-notation-time-and-depth)
3. [Why `return_sequences=True` Matters](#3-why-return_sequencestrue-matters)
4. [Building It in Keras: Architecture and Real Parameter Counts](#4-building-it-in-keras-architecture-and-real-parameter-counts)
5. [When to Use a Deep RNN](#5-when-to-use-a-deep-rnn)
6. [Deep LSTM and Deep GRU](#6-deep-lstm-and-deep-gru)
7. [Disadvantages: Overfitting and Training Time](#7-disadvantages-overfitting-and-training-time)
8. [Complete Verified Code](#8-complete-verified-code)
9. [Key Takeaways](#9-key-takeaways)
10. [Further Reading](#10-further-reading)

---

## 1. From One Layer to Many

A single-layer RNN unrolls in one dimension: time. Stacking multiple RNN layers adds a second dimension: depth. Each layer's full output *sequence* becomes the next layer's input sequence, so information flows both horizontally (across time steps, within a layer) and vertically (across layers, within a time step).

![Single vs deep RNN](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_single_vs_deep_rnn.png)

*Left: a single-layer RNN — one row of cells, connected only through time (red arrows). Right: a 3-layer deep RNN — the same time connections exist within each row, but now every cell also feeds the cell directly above it (blue arrows) at the same time step. The result is a 2D grid: time runs left-to-right, depth runs bottom-to-top. Layer 1 might learn low-level patterns (e.g. local word combinations), while layer 3 can combine those into higher-level patterns (e.g. sentence-level sentiment) — the same hierarchical idea as stacking layers in a CNN.*

---

## 2. Notation: Time and Depth

With two dimensions to track, each hidden state needs two indices: a time index `t` and a depth (layer) index `l`. The general formula for any cell in the stack takes input from two places — the same layer's previous time step, and the layer below at the same time step.

![Notation](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_notation_time_depth.png)

*`h_t^(l)` is the hidden state at time `t`, layer `l`. It's computed from `h_(t-1)^(l)` (red arrows — the recurrent connection within this layer) and `h_t^(l-1)` (blue arrows — the output of the layer directly below, at the same time step). The base case `h_t^(0) = x_t` says layer 0 is just the raw input — so layer 1's "input from below" is simply the sequence itself. This notation is identical whether the cell `f` is a SimpleRNN, LSTM, or GRU cell; only what happens inside `f` changes.*

---

## 3. Why `return_sequences=True` Matters

In Keras, a recurrent layer by default returns only its *last* hidden state — a single vector, not a sequence. That's fine for the final layer in a stack (whose output feeds a Dense layer), but it breaks any layer that needs to feed a *sequence* into the layer above it.

![return_sequences](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_return_sequences.png)

*Left (wrong): without `return_sequences=True` on layer 1, only its last hidden state is produced — layer 2 receives a single vector, not a time series, so it has nothing to recur over. Right (correct): with `return_sequences=True` on every layer except the last, each layer passes its full sequence of hidden states up to the next layer, preserving the time dimension throughout the stack.*

I confirmed this is a real, enforced constraint by removing `return_sequences=True` from the first layer and building the model in Keras 3. It fails immediately with:

```
ValueError: Input 0 with name 'None' of layer 'simple_rnn_1' is incompatible
with the layer: expected ndim=3, found ndim=2. Full shape received: (None, 5)
```

Keras caught the exact mistake the diagram shows: layer 2 expected a 3D sequence input (batch, time, features) but only got a 2D single-vector input (batch, features), because layer 1 had already collapsed the time dimension.

---

## 4. Building It in Keras: Architecture and Real Parameter Counts

All three variants from your code — stacked SimpleRNN, stacked LSTM, stacked GRU — share the same four-layer structure: an Embedding layer, two recurrent layers (`return_sequences=True` on the first, default on the second), and a Dense output layer.

![Architecture and parameters](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_architecture_and_params.png)

*Left: the shared architecture — only the recurrent cell type changes between the three models. Right: real parameter counts, taken from running `model.summary()` in Keras 3 for each variant with `Embedding(10000, 32)`, two recurrent layers of 5 units each, and `Dense(1)`, on inputs padded to length 100. The Embedding layer dominates every model at 320,000 parameters (10,000 words x 32 dimensions) — the recurrent layers are tiny by comparison. Within the recurrent layers, the ordering holds exactly as expected from the gate counts in the earlier notes: LSTM (4 gates worth of weights) > GRU (3 gates worth) > SimpleRNN (no gates), both for layer 1 (760 > 585 > 190) and layer 2 (220 > 180 > 55).*

Total parameters per model (Embedding + both recurrent layers + Dense, real `model.summary()` output):

| Model | Total parameters |
|---|---|
| Deep SimpleRNN | 320,251 |
| Deep LSTM | 320,986 |
| Deep GRU | 320,771 |

---

## 5. When to Use a Deep RNN

![Usage scenarios](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_usage_and_overfitting.png)

*Left: the three conditions worth having in place before reaching for a deep RNN — a genuinely complex task (speech recognition, machine translation), a large enough dataset to make the extra parameters worth learning, and enough compute budget for longer training runs. Right: an **illustrative** sketch (not measured data) of the kind of overfitting pattern that can show up with deep RNNs — training loss keeps falling while validation loss flattens out and then rises, which is the signal to add regularization or stop training early.*

---

## 6. Deep LSTM and Deep GRU

Everything in Sections 1-4 applies unchanged if you swap the `SimpleRNN` cells for `LSTM` or `GRU` cells — the stacking logic (time + depth, `return_sequences=True` between layers) doesn't depend on what's inside the cell. In practice, deep LSTMs and deep GRUs are more common than deep SimpleRNNs, because stacking makes the vanishing/exploding gradient problem from [backpropagation-through-time.md](backpropagation-through-time.md) worse — more layers means more opportunities for gradients to shrink or blow up — and LSTM/GRU cells (from [lstm-introduction.md](lstm-introduction.md) and [gru-gated-recurrent-units.md](gru-gated-recurrent-units.md)) were built specifically to resist that.

---

## 7. Disadvantages: Overfitting and Training Time

Deep RNNs trade simplicity for representational power, and that trade has real costs:

- **More parameters to learn** means a higher risk of overfitting, especially on smaller datasets — the model can memorize training sequences instead of learning generalizable patterns.
- **Longer training time** follows directly from having more layers and more weights to update every step, compounded by the fact that recurrent layers can't be parallelized across time the way convolutional layers can.
- **Careful regularization and tuning become more important**: dropout (including recurrent dropout), smaller hidden sizes, early stopping, and more deliberate learning-rate choices all matter more as depth increases.

---

## 8. Complete Verified Code

Your code was tested directly in Keras 3.15 / TensorFlow 2.21. Two small fixes were needed: `input_length` is accepted without error in this Keras version but silently leaves every layer unbuilt (so `model.summary()` shows "unbuilt" with 0 params everywhere) — using an explicit `Input(shape=(100,))` layer instead builds the model properly and gives real parameter counts.

```python
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, LSTM, GRU, Dense

# Load the IMDb dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Pad sequences to have the same length
x_train = pad_sequences(x_train, maxlen=100)
x_test = pad_sequences(x_test, maxlen=100)

def build_model(cell):
    return Sequential([
        Input(shape=(100,)),
        Embedding(10000, 32),
        cell(5, return_sequences=True),   # must return the full sequence...
        cell(5),                          # ...so this second layer can recur over it
        Dense(1, activation='sigmoid'),
    ])

# Deep (stacked) RNN / LSTM / GRU — same structure, different cell
rnn_model  = build_model(SimpleRNN); rnn_model.summary()
lstm_model = build_model(LSTM);      lstm_model.summary()
gru_model  = build_model(GRU);       gru_model.summary()

for model in (rnn_model, lstm_model, gru_model):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

I did not run `model.fit()` on the IMDB data (no training or accuracy numbers are reported here) — only the model construction and `model.summary()` calls were executed, which is what produced the real parameter counts in Section 4.

---

## 9. Key Takeaways

- A Deep (Stacked) RNN adds a second, vertical dimension — depth — on top of the usual time dimension, by feeding each layer's output sequence into the next layer.
- Each cell is indexed by both time `t` and depth `l`: `h_t^(l) = f(W_x^(l) h_t^(l-1) + W_h^(l) h_(t-1)^(l) + b^(l))`, with the base case `h_t^(0) = x_t`.
- In Keras, every stacked recurrent layer except the last needs `return_sequences=True`, or the next layer receives a single vector instead of a sequence and the model fails to build — confirmed directly with a real Keras error.
- Across SimpleRNN, LSTM, and GRU versions of the same architecture, the Embedding layer dominates the parameter count (320,000 of roughly 320,250-320,986 total); the gate-count ordering from earlier notes (LSTM > GRU > SimpleRNN) carries over exactly to the stacked recurrent layers.
- Deep RNNs are best reserved for complex tasks (speech recognition, machine translation) with enough data and compute to justify the extra parameters.
- Everything about stacking (notation, `return_sequences`, depth) applies identically to Deep LSTMs and Deep GRUs, which are more common in practice because their gating helps counteract the extra vanishing/exploding-gradient risk that comes with more layers.
- More layers means more overfitting risk and longer training time, making regularization and careful tuning more important than in a single-layer model.

---

## 10. Further Reading

- Graves, A., Mohamed, A., & Hinton, G. (2013). *Speech Recognition with Deep Recurrent Neural Networks*. ICASSP. One of the influential papers demonstrating stacked RNNs for speech recognition.
- Pascanu, R., Gulcehre, C., Cho, K., & Bengio, Y. (2014). *How to Construct Deep Recurrent Neural Networks*. ICLR. Formalizes several ways of adding depth to RNNs, including the stacking approach covered here.
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). *Sequence to Sequence Learning with Neural Networks*. NeurIPS. Uses a 4-layer deep LSTM for machine translation.
- Maas, A. L., et al. (2011). *Learning Word Vectors for Sentiment Analysis*. ACL. Source of the IMDB dataset used in the code example.
