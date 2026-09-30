# Implementing RNNs in Keras for Sentiment Analysis (IMDB)

This note walks through the full workflow for building a Recurrent Neural Network in Keras that classifies movie reviews as positive or negative: turning raw text into integer sequences, padding them to a fixed length, learning word embeddings, and stacking `Embedding → SimpleRNN → Dense(sigmoid)`. It builds on the concepts in [recurrent-neural-networks-intro.md](recurrent-neural-networks-intro.md) (why sequences need memory, the hidden-state recurrence) and turns them into working code. The goal, as in the lecture, is conceptual understanding rather than state-of-the-art accuracy.

## Table of Contents
1. [Text Preprocessing](#1-text-preprocessing)
2. [Padding and Truncation](#2-padding-and-truncation)
3. [Embeddings vs Integer Encoding](#3-embeddings-vs-integer-encoding)
4. [Model Architecture](#4-model-architecture)
5. [How the SimpleRNN Layer Processes a Review](#5-how-the-simplernn-layer-processes-a-review)
6. [Formulas](#6-formulas)
7. [Your Code, Explained and Corrected](#7-your-code-explained-and-corrected)
8. [Complete Working Code](#8-complete-working-code)
9. [Key Takeaways](#9-key-takeaways)
10. [Further Reading](#10-further-reading)

---

## 1. Text Preprocessing

Neural networks consume numbers, not strings, so raw text goes through three steps:

1. **Tokenization**: split each review into tokens (words).
2. **Integer encoding**: map every token to an integer id using a vocabulary. The Keras IMDB dataset ships already integer-encoded, with `0` = padding, `1` = start-of-sequence, `2` = unknown, and real words starting at `3`.
3. **Padding**: force every sequence to the same length (Section 2).

![Text preprocessing pipeline](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_preprocessing_pipeline.png)

*The four boxes show one toy review moving through the pipeline: the raw string "this movie was great", its token list, its integer encoding `[3, 4, 5, 6]` (from a small toy vocabulary, not the real IMDB one), and finally the version padded with 46 zeros to length 50. Every review in the dataset goes through the same steps so that a batch can be stacked into one rectangular array.*

---

## 2. Padding and Truncation

A batch must be a rectangular tensor, but reviews have different lengths. `pad_sequences(..., maxlen=50)` pads short reviews and truncates long ones. With `padding='post'` zeros are added at the end. By default (`truncating='pre'`) a long review loses its **beginning**, keeping the last 50 tokens.

![Padding and truncation](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_padding_and_truncation.png)

*Top: a 4-token review padded with 46 zeros at the end. Bottom: a 120-token review truncated to its final 50 tokens. Note that the zeros are ordinary inputs, not skipped steps: unless you set `mask_zero=True` on the Embedding layer, the RNN runs over them like any other token. I confirmed the padding behaviour by running `pad_sequences` directly: `[5,9,2]` with `padding='post', maxlen=6` gives `[5 9 2 0 0 0]`, and a 7-token sequence with `maxlen=5` keeps the last five tokens.*

**Why post-padding matters for SimpleRNN.** The classifier only reads the *final* hidden state. With `padding='post'` that final state comes after dozens of zero inputs. In a real run on the toy review `[3,4,5,6]` with untrained weights (seed 42), the cosine similarity between the hidden state right after the last real word (`h4`) and the final state `h50` was **0.142**, while with `padding='pre'` (zeros first, real words last) the final state stayed much closer to `h4` at **0.545**. This is one reason `padding='pre'` is often preferred with simple RNNs. These are untrained weights on one toy input, so treat the numbers as a demonstration of the mechanism, not a measured accuracy difference.

---

## 3. Embeddings vs Integer Encoding

Integer ids are arbitrary labels: the fact that `great = 6` and `was = 5` differ by 1 means nothing. An **Embedding layer** replaces each id with a dense, trainable vector. It is literally a lookup table of shape `(vocab_size, embedding_dim)`; row `k` is the vector for word id `k`.

![Integer encoding vs embedding lookup](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_integer_vs_embedding_lookup.png)

*Left: integer encoding gives each word an id whose numeric distance carries no meaning. Right: `Embedding(10000, 2)` looks up rows of a 10000 x 2 matrix. The four vectors shown are the real values of rows 3 to 6 from an untrained run with seed 42, so they are small random numbers; training adjusts them by backpropagation until words used in similar ways get similar vectors.*

![Illustrative embedding space](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_embedding_space_illustrative.png)

*This plot is **illustrative only**: the points are hand-placed to show the idea that, after training, positive words ("great", "loved", "amazing") can cluster together, negative words ("terrible", "boring", "awful") cluster elsewhere, and neutral words ("movie", "film") sit between. It is not measured from a trained model.*

The lecture's point is that learned embeddings capture semantic relationships and typically give better results than raw integer encoding. Your code uses a 2-dimensional embedding, which is deliberately tiny so the model stays easy to inspect; real models use dimensions like 50 to 300.

---

## 4. Model Architecture

The model is `Embedding → SimpleRNN(32) → Dense(1, sigmoid)`.

![Model architecture](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_model_architecture.png)

*Tensor shapes flow left to right: a batch of integer sequences `(None, 50)` becomes `(None, 50, 2)` after the embedding, a single `(None, 32)` vector after the SimpleRNN (because `return_sequences=False` keeps only the last hidden state), and a `(None, 1)` probability after the sigmoid Dense layer. The parameter counts are the real output of `model.summary()` in Keras 3: 20,000 + 1,120 + 33 = 21,153.*

![Parameter counts](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/09_parameter_counts.png)

*Almost all parameters (20,000 of 21,153) live in the embedding table, because it stores 2 numbers for each of 10,000 vocabulary words. The recurrent layer is small (1,120) because its weights are shared across all 50 time steps.*

Abridged `model.summary()` output for the corrected model (layer numbering in the names may differ on your machine):

```
Layer (type)                    Output Shape           Param #
embedding (Embedding)           (None, 50, 2)           20,000
simple_rnn (SimpleRNN)          (None, 32)               1,120
dense (Dense)                   (None, 1)                   33
Total params: 21,153 (82.63 KB)
```

---

## 5. How the SimpleRNN Layer Processes a Review

The layer applies the same cell 50 times, once per token, carrying a 32-dimensional hidden state forward. Only the last state is passed on.

![Unrolled RNN](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_rnn_unrolled_last_state.png)

*The RNN is unrolled over 50 steps for the padded review "this movie was great". The red arrows are the hidden state moving forward in time. Steps 1 to 4 read real words; steps 5 to 50 read padding zeros (grey). Only the hidden state at step 50 is sent to the Dense layer, which is why the position of the padding matters.*

To confirm that I understand what Keras is doing, I extracted the layer's weights and re-implemented the forward pass in plain NumPy. Over all 50 time steps and 32 units, the largest absolute difference between Keras's hidden states and my NumPy hidden states was about **5.5e-08**, which is floating-point rounding. The final probability for the toy review was **0.5012** from Keras and **0.5012** from NumPy. That value is near 0.5 because the weights are untrained.

![Hidden-state evolution](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_hidden_state_evolution.png)

*Left: the real 32-unit hidden state at each of the 50 steps (untrained weights); the dashed line marks where the padding begins. Right: the magnitude of the hidden state over time. The state keeps changing during the padding steps instead of freezing, which shows that the RNN really does process the zeros.*

---

## 6. Formulas

Embedding lookup for token id `k`: `x_t = E[k]`, where `E` has shape `(10000, 2)`.

SimpleRNN recurrence (`h_0 = 0`):

```
h_t = tanh(x_t · W_xh + h_(t-1) · W_hh + b)
```

Here `W_xh` has shape `(2, 32)`, `W_hh` has shape `(32, 32)` and `b` has shape `(32,)`.

Output layer using the last state only:

```
z = h_50 · W_d + b_d
p = sigmoid(z) = 1 / (1 + e^(-z))
```

Parameter count of a SimpleRNN layer: `units × (input_dim + units + 1) = 32 × (2 + 32 + 1) = 1,120`. Embedding: `10,000 × 2 = 20,000`. Dense: `32 × 1 + 1 = 33`.

Loss for binary classification: `L = -[y·log(p) + (1-y)·log(1-p)]`.

![Sigmoid output](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/08_sigmoid_output.png)

*The sigmoid squashes the score `z` into (0, 1). Outputs above 0.5 are read as positive sentiment and outputs below 0.5 as negative, which is why the network needs just one output unit for a two-class problem.*

---

## 7. Your Code, Explained and Corrected

Your snippet is close, but running it in Keras 3 (I tested with Keras 3.15.1) shows a few things to fix:

- **`X_train` is never defined.** You need to load the data first: `(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)`. The `num_words=10000` must match the `10000` in the Embedding layer.
- **`Embedding(10000, 2, 50)` raises an error.** The third positional argument of `Embedding` is the initializer, not the sequence length, so Keras 3 fails with `ValueError: Could not interpret initializer identifier: 50`. The old `input_length` argument was removed in Keras 3; instead give the model its input shape with `Input(shape=(50,))` (or `model.build((None, 50))`).
- **`from keras.preprocessing.text import Tokenizer` fails in Keras 3.15** (`No module named 'keras.preprocessing.text'`). You do not need it here because IMDB is already integer-encoded, so remove it. `Flatten` is also imported but unused.
- `model.summary()` then works and gives the 21,153-parameter table shown above.

---

## 8. Complete Working Code

```python
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras import Sequential, Input
from keras.layers import Dense, SimpleRNN, Embedding

# 1. Load integer-encoded reviews (vocabulary limited to 10,000 words)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# 2. Pad / truncate every review to 50 tokens
X_train = pad_sequences(X_train, padding='post', maxlen=50)
X_test  = pad_sequences(X_test,  padding='post', maxlen=50)

# 3. Build the model
model = Sequential([
    Input(shape=(50,)),
    Embedding(10000, 2),                      # 10000 x 2 lookup table
    SimpleRNN(32, return_sequences=False),    # keep only the last hidden state
    Dense(1, activation='sigmoid'),
])
model.summary()

# 4. Train and evaluate
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
print(model.evaluate(X_test, y_test))
```

Status of this code: the model definition and `model.summary()` were run and produce the output above. The IMDB download and the training step (`load_data`, `fit`, `evaluate`) could not be run in my environment because the dataset host was not reachable, so no accuracy figures are quoted in these notes. Run it locally to see your own numbers.

The same recurrence from scratch in NumPy (this matched Keras to about 5.5e-08 when I ran it with the same weights):

```python
import numpy as np

def simple_rnn_forward(token_ids, emb, W_xh, W_hh, b):
    h = np.zeros(W_hh.shape[0])
    states = []
    for t in token_ids:
        x = emb[t]                            # embedding lookup
        h = np.tanh(x @ W_xh + h @ W_hh + b)  # recurrence
        states.append(h)
    return np.array(states)                   # last row = input to Dense
```

---

## 9. Key Takeaways

- Text must be tokenized, integer-encoded and padded to a fixed length before an RNN can use it.
- `pad_sequences` pads with zeros and, by default, truncates from the front; the RNN treats padding zeros as real inputs unless masking is enabled.
- Because the classifier reads only the **last** hidden state, `padding='post'` forces the network to remember the review across many zero steps; `padding='pre'` is often kinder to a SimpleRNN.
- Embedding layers replace arbitrary integer ids with learned dense vectors and typically outperform raw integer encoding.
- The model here has 21,153 parameters: 20,000 in the embedding, 1,120 in the SimpleRNN, 33 in the Dense layer.
- In Keras 3, `Embedding` no longer takes `input_length`; declare the input shape with `Input(...)`.

---

## 10. Further Reading

- Maas, A. L., et al. (2011). *Learning Word Vectors for Sentiment Analysis*. ACL. Introduces the IMDB movie review dataset.
- Mikolov, T., et al. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv:1301.3781. Word embeddings (word2vec).
- Elman, J. L. (1990). *Finding Structure in Time*. Cognitive Science, 14(2). The simple recurrent network.
- Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning. Keras-based treatment of text and sequence models.
